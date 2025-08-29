import os
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from copy import deepcopy
from utils.buffer import Buffer
from utils.args import *
from models.utils.continual_model import ContinualModel
from backbone.MNISTMLP import SparseMNISTMLP
from backbone.SparseResNet18 import sparse_resnet18
from models.utils.losses import SupConLoss
from models.utils.pos_groups import class_dict, pos_groups


num_classes_dict = {
    'seq-cifar10': 10,
    'seq-cifar100': 100,
    'seq-tinyimg': 200,
    'gcil-cifar100': 100,
    'seq-imagenet100': 100,
}


def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description='Semantic Aware Representation Learning')
    add_management_args(parser)
    add_experiment_args(parser)
    add_rehearsal_args(parser)
    # Consistency Regularization Weight
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--op_weight', type=float, default=0.1)
    parser.add_argument('--sim_thresh', type=float, default=0.80)
    parser.add_argument('--sm_weight', type=float, default=0.01)
    # Sparsity param
    parser.add_argument('--apply_kw', nargs='*', type=int, default=[1, 1, 1, 1])
    parser.add_argument('--kw', type=float, nargs='*', default=[0.9, 0.9, 0.9, 0.9])
    parser.add_argument('--kw_relu', type=int, default=1)
    parser.add_argument('--kw_local', type=int, default=1)
    parser.add_argument('--num_feats', type=int, default=512)
    # Experimental Args
    parser.add_argument('--save_interim', type=int, default=1)
    parser.add_argument('--warmup_epochs', type=int, default=5)
    parser.add_argument('--use_lr_scheduler', type=int, default=1)
    parser.add_argument('--lr_steps', type=int, nargs='*', default=[70, 90])

    parser.add_argument('--cc', type=int, default=1)
    parser.add_argument('--cov_weight', type=float, default=0.1)
    parser.add_argument('--cov_shrink_alpha', type=float, default=0.5)
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SARLMACIL(ContinualModel):
    NAME = 'sarl_macil'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLMACIL, self).__init__(backbone, loss, args, transform)
        self.buffer = Buffer(self.args.buffer_size, self.device)

        # Initialize plastic and stable model
        if 'mnist' in self.args.dataset:
            self.net = SparseMNISTMLP(28 * 28, 10, kw_percent_on=args.kw).to(self.device)
        else:
            self.net = sparse_resnet18(
                nclasses=num_classes_dict[args.dataset],
                kw_percent_on=args.kw, local=args.kw_local,
                relu=args.kw_relu, apply_kw=args.apply_kw
            ).to(self.device)

        self.net_old = None
        self.get_optimizer()

        # set regularization weight
        self.alpha = args.alpha

        self.consistency_loss = nn.MSELoss(reduction='none')
        self.current_task = 0
        self.global_step = 0
        self.lst_models = ['net']

        # init Object Prototypes
        self.op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.op_sum = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.running_op = torch.zeros(num_classes_dict[args.dataset], args.num_feats).to(self.device)
        self.running_sample_counts = torch.zeros(num_classes_dict[args.dataset]).to(self.device)

        self.learned_classes = []
        self.flag = True
        self.eval_prototypes = True
        self.pos_groups = {}
        self.dist_mat = torch.zeros(num_classes_dict[args.dataset], num_classes_dict[args.dataset]).to(self.device)
        self.class_dict = class_dict[args.dataset]

    # -------------------------
    # Covariance Calibration utils
    # -------------------------
    def _shrink_cov(self, cov, alpha=0.5):
        """
        Simple Ledoit-Wolf-ish shrinkage toward diagonal/identity:
        cov: torch.Tensor (D,D)
        alpha: shrinkage strength in [0,1]
        returns: shrunk cov (D,D)
        """
        # ensure double precision for stability
        cov = cov.double()
        diag = torch.diag(torch.diagonal(cov)).to(cov.device)
        shrunk = (1 - alpha) * cov + alpha * diag
        # add small eps to diagonal for numerical stability
        shrunk = shrunk + torch.eye(shrunk.size(0), device=shrunk.device, dtype=shrunk.dtype) * 1e-6
        return shrunk.float()

    def _compute_buffer_class_stats(self):
        """
        Compute class means and inverse covariances from the rehearsal buffer using self.net_old.
        Store:
          - self.class_means: tensor (C, D)
          - self.class_invcovs: tensor (C, D, D)
        Only compute for classes in self.learned_classes.
        """
        if self.net_old is None or self.buffer.is_empty():
            return

        # get all buffer data
        buf_inputs, buf_labels, _ = self.buffer.get_all_data(transform=self.transform)
        # create loader
        buf_ds = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        loader = DataLoader(buf_ds, batch_size=self.args.batch_size, shuffle=False, num_workers=0)

        # collect per-class features using net_old
        feat_dict = {}
        with torch.no_grad():
            self.net_old.eval()
            for inputs, labels in loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, activations = self.net_old(inputs, return_activations=True)
                feats = F.normalize(activations['feat']).detach().cpu()
                labels_cpu = labels.cpu()
                for i, lab in enumerate(labels_cpu):
                    lab = int(lab.item())
                    if lab not in feat_dict:
                        feat_dict[lab] = []
                    feat_dict[lab].append(feats[i])

        # build class_means & covs
        num_classes = num_classes_dict[self.args.dataset]
        D = self.op.size(1)

        self.class_means = torch.zeros((num_classes, D), device=self.device)
        self.class_invcovs = torch.zeros((num_classes, D, D), device=self.device)

        for c in self.learned_classes:
            if c not in feat_dict or len(feat_dict[c]) == 0:
                self.class_means[c] = self.op[c].detach()
                self.class_invcovs[c] = torch.eye(D, device=self.device)
                continue

            feats_c = torch.stack(feat_dict[c], dim=0)          # (Nc, D) on CPU
            mean_c = feats_c.mean(dim=0).to(self.device)        # (D,)

            X = feats_c.double()                                # (Nc, D) float64
            if X.size(0) >= 2:
                cov_c = torch.cov(X.T)                          # (D, D), float64
            else:
                var_diag = X.var(dim=0, unbiased=False)         # (D,)
                cov_c = torch.diag(var_diag)                    # (D, D)

            cov_c = cov_c.to(self.device).float()

            cov_c = self._shrink_cov(cov_c, alpha=self.args.cov_shrink_alpha)
            try:
                invcov = torch.linalg.pinv(cov_c)
            except Exception:
                invcov = torch.eye(D, device=self.device)

            self.class_means[c] = mean_c
            self.class_invcovs[c] = invcov

    def _mahalanobis_loss_from_buffer(self):
        """
        Compute Mahalanobis loss on buffer data using stored class_means and class_invcovs.
        Return scalar loss (torch.Tensor).
        We compute per-sample (x - mu_c)^T invcov_c (x - mu_c) for the true class c, and average.
        """
        if self.buffer.is_empty() or not hasattr(self, 'class_invcovs') or not hasattr(self, 'class_means'):
            return torch.tensor(0.0, device=self.device)

        buf_inputs, buf_labels, _ = self.buffer.get_all_data(transform=self.transform)
        buf_ds = torch.utils.data.TensorDataset(buf_inputs, buf_labels)
        loader = DataLoader(buf_ds, batch_size=self.args.minibatch_size, shuffle=False, num_workers=0)

        total_loss = 0.0
        total_cnt = 0
        with torch.no_grad():  # features can be computed without grad (we compute loss with grad later?)
            # IMPORTANT: we need features from current net (so grad flows). So do not wrap entire loop in no_grad.
            pass

        # we compute features with grad enabled so the Mahalanobis loss will backprop into net's params
        for inputs, labels in loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            outputs, activations = self.net(inputs, return_activations=True)
            feats = F.normalize(activations['feat'])  # (B, D)

            batch_loss = torch.tensor(0.0, device=self.device)
            valid_cnt = 0
            for i, lab in enumerate(labels):
                lab_i = int(lab.item())
                if lab_i >= self.class_means.size(0):
                    # skip if no stats
                    continue
                mu = self.class_means[lab_i]  # (D,)
                invcov = self.class_invcovs[lab_i]  # (D,D)
                diff = (feats[i] - mu).unsqueeze(0)  # (1, D)
                # (1,D) @ (D,D) -> (1,D) ; then @ (D,1) -> scalar
                m = diff @ invcov @ diff.t()
                batch_loss = batch_loss + m.squeeze()
                valid_cnt += 1

            if valid_cnt > 0:
                total_loss += batch_loss
                total_cnt += valid_cnt

        if total_cnt == 0:
            return torch.tensor(0.0, device=self.device)
        return (total_loss / float(total_cnt))

    def observe(self, inputs, labels, not_aug_inputs):
        real_batch_size = inputs.shape[0]
        self.opt.zero_grad()
        self.net.train()
        loss = 0
        if not self.buffer.is_empty():
            buf_inputs, buf_labels, buf_logits = self.buffer.get_data(self.args.minibatch_size, transform=self.transform)
            buff_out, buff_activations = self.net(buf_inputs, return_activations=True)
            buff_feats = buff_activations['feat']
            reg_loss = self.args.alpha * F.mse_loss(buff_out, buf_logits)

            buff_ce_loss = self.loss(buff_out, buf_labels)
            loss += reg_loss + buff_ce_loss

            # ---------------------------
            # Covariance Calibration Loss
            # ---------------------------
            if getattr(self.args, 'cc', 0) == 1 and self.current_task > 0:
                cov_loss = self._mahalanobis_loss_from_buffer()
                # cov_loss is average Mahalanobis scalar; add weighted to main loss (no detach)
                loss = loss + self.args.cov_weight * cov_loss
                if hasattr(self, 'writer'):
                    try:
                        self.writer.add_scalar(f'Task {self.current_task}/cov_loss', cov_loss.item(), self.iteration)
                    except Exception:
                        pass

            # Regularization loss on Class Prototypes
            if self.current_task > 0:
                buff_feats = F.normalize(buff_feats)
                dist = 0
                for class_label in torch.unique(buf_labels):
                    if class_label in self.learned_classes:
                        image_class_mask = (buf_labels == class_label)
                        mean_feat = buff_feats[image_class_mask].mean(axis=0)
                        dist += F.mse_loss(mean_feat, self.op[class_label])

                loss += self.args.op_weight * dist

            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/reg_loss', reg_loss.item(), self.iteration)
                self.writer.add_scalar(f'Task {self.current_task}/buff_ce_loss', buff_ce_loss.item(), self.iteration)

        outputs, activations = self.net(inputs, return_activations=True)

        if self.epoch > self.args.warmup_epochs and self.current_task > 0:

            outputs_old = self.net_old(inputs)
            loss += self.args.beta * F.mse_loss(outputs, outputs_old)

            new_labels = [i.item() for i in torch.unique(labels) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels
            feats = activations['feat']
            feats = F.normalize(feats)

            # Accumulate the class prototypes
            class_prot = {}
            for ref_class_label in self.learned_classes:
                class_prot[ref_class_label] = self.op[ref_class_label]
            for class_label in new_labels:
                class_prot[class_label] = feats[labels == class_label].mean(dim=0)

            l_cont = 0
            for class_label in new_labels:
                pos_dist = 0
                neg_dist = 0
                for ref_class_label in all_labels:
                    if class_label != ref_class_label:
                        if ref_class_label in self.pos_groups[class_label]:
                            pos_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])
                        else:
                            neg_dist += F.mse_loss(class_prot[class_label], class_prot[ref_class_label])

                if neg_dist > 0:
                    l_cont += pos_dist/neg_dist

            loss += self.args.sm_weight * l_cont
            if hasattr(self, 'writer'):
                self.writer.add_scalar(f'Task {self.current_task}/contrastive_loss', l_cont.item(), self.iteration)

        ce_loss = self.loss(outputs, labels)
        loss += ce_loss

        if torch.isnan(loss):
            raise ValueError('NAN Loss')

        # Log values
        if hasattr(self, 'writer'):
            self.writer.add_scalar(f'Task {self.current_task}/ce_loss', ce_loss.item(), self.iteration)
            self.writer.add_scalar(f'Task {self.current_task}/loss', loss.item(), self.iteration)

        loss.backward()
        self.opt.step()

        self.buffer.add_data(
            examples=not_aug_inputs,
            labels=labels[:real_batch_size],
            logits=outputs.data,
        )

        return loss.item()

    def end_epoch(self, dataset, epoch) -> None:

        if self.scheduler is not None:
            self.scheduler.step()

        self.flag = True
        self.net.eval()

        # Calculate the Class Prototypes and Covariance Matrices using Working Model
        if self.epoch >= self.args.warmup_epochs and self.eval_prototypes and self.current_task > 0:
            print('!' * 30)
            print('Evaluating Prototypes for the New Classes')
            # Calculate CLass Prototypes
            X = []
            Y = []
            for data in dataset.train_loader:
                inputs, labels, not_aug_inputs = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                # print(input.shape)
                outputs, activations = self.net(inputs, return_activations=True)
                feat = activations['feat']

                # Normalize Features
                feat = F.normalize(feat)  # Is it needed

                X.append(feat.detach().cpu().numpy())
                Y.append(labels.cpu().numpy())

                unique_labels = labels.unique()
                for class_label in unique_labels:
                    self.running_op[class_label] += feat[labels == class_label].sum(dim=0).detach()
                    self.running_sample_counts[class_label] += (labels == class_label).sum().detach()

            X = np.concatenate(X, axis=0)
            Y = np.concatenate(Y, axis=0)

            # Take average feats
            for class_label in np.unique(Y):
                self.running_op[class_label] = self.running_op[class_label] / self.running_sample_counts[class_label]

            # Calculate Covariance Matrix
            class_mean_set = []
            for class_label in np.unique(Y):

                image_class_mask = (Y == class_label)
                class_mean_set.append(np.mean(X[image_class_mask], axis=0))

            # Evaluate the distances
            new_labels = [i for i in np.unique(Y) if i not in self.learned_classes]
            all_labels = self.learned_classes + new_labels

            cos = nn.CosineSimilarity(dim=0, eps=1e-6)

            # dist_mat = torch.zeros(len(all_labels), len(all_labels))
            for class_label in new_labels:
                for ref_class_label in new_labels:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.running_op[ref_class_label])
                for ref_class_label in self.learned_classes:
                    self.dist_mat[class_label, ref_class_label] = cos(self.running_op[class_label], self.op[ref_class_label])

            print('*' * 30)
            print('Positive Groups')
            for class_label in new_labels:
                pos_group = self.dist_mat[class_label] > self.args.sim_thresh
                self.pos_groups[class_label] = [i for i in all_labels if pos_group[i]]
                if self.args.dataset not in ['seq-tinyimg', 'gcil-cifar100']:
                    print(f'{self.class_dict[class_label]}: ' + ', '.join([self.class_dict[i] for i in self.pos_groups[class_label]]))
                else:
                    print(f'{class_label}:', self.pos_groups[class_label])
            print('*' * 30)

            self.eval_prototypes = False

    def end_task(self, dataset) -> None:

        # reset optimizer
        self.get_optimizer()

        self.eval_prototypes = True
        self.flag = True
        self.current_task += 1
        self.net.eval()

        # Save old model
        self.net_old = deepcopy(self.net)
        self.net_old.eval()

        # If covariance calibration enabled, compute class means & inverse covariances from buffer (using net_old)
        if getattr(self.args, 'cc', 0) == 1 and not self.buffer.is_empty():
            print("Computing class statistics (means + invcov) from buffer using net_old for CC...")
            # ensure learned_classes is non-empty; if buffer has classes not in learned_classes, include them
            # (we assume learned_classes is up-to-date)
            self._compute_buffer_class_stats()
            print("Done computing class statistics for CC.")

        # =====================================
        # Buffer Pass
        # =====================================
        buf_inputs, buf_labels, buf_logits = self.buffer.get_all_data(transform=self.transform)
        buf_idx = torch.arange(0, len(buf_labels)).to(buf_labels.device)

        buff_dataset = torch.utils.data.TensorDataset(buf_inputs, buf_labels, buf_logits, buf_idx)
        buff_data_loader = DataLoader(buff_dataset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        self.net.train()
        for data, label, logits, index in buff_data_loader:
            out_net = self.net(data)

        # =====================================
        # Calculate CLass Prototypes
        # =====================================
        self.net.eval()
        X = []
        Y = []
        for data in dataset.train_loader:
            inputs, labels, not_aug_inputs = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            # print(input.shape)
            outputs, activations = self.net(inputs, return_activations=True)
            feat = activations['feat']

            # Normalize Features
            feat = F.normalize(feat)  # Is it needed

            X.append(feat.detach().cpu().numpy())
            Y.append(labels.cpu().numpy())

            unique_labels = labels.unique()
            for class_label in unique_labels:
                self.op_sum[class_label] += feat[labels == class_label].sum(dim=0).detach()
                self.sample_counts[class_label] += (labels == class_label).sum().detach()

        X = np.concatenate(X)
        Y = np.concatenate(Y)

        # Take average feats
        for class_label in np.unique(Y):
            if class_label not in self.learned_classes:
                self.learned_classes.append(class_label)
            self.op[class_label] = self.op_sum[class_label] / self.sample_counts[class_label]

        if self.args.save_interim:
            model_dir = os.path.join(self.args.output_dir, "task_models", dataset.NAME, self.args.experiment_id)
            os.makedirs(model_dir, exist_ok=True)
            torch.save(self.net, os.path.join(model_dir, f'task{self.current_task}'))
            torch.save(self.op, os.path.join(model_dir, f'object_ptototypes.ph'))

    def get_optimizer(self):
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
