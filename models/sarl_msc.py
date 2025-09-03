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

    parser.add_argument('--msc_damp_factor', type=float, default=0.1,
                        help='Factor to dampen the calculated shift vector.')
    return parser


# =============================================================================
# Mean-ER
# =============================================================================
class SARLMSC(ContinualModel):
    NAME = 'sarl_msc'
    COMPATIBILITY = ['class-il', 'domain-il', 'task-il', 'general-continual']

    def __init__(self, backbone, loss, args, transform):
        super(SARLMSC, self).__init__(backbone, loss, args, transform)
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

        self.target_prototypes = None

    def _extract_features_with_labels(self, data_loader, model):
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for data_batch in data_loader:
                if len(data_batch) == 3:
                    inputs, labels, _ = data_batch
                else:
                    inputs, labels = data_batch

                inputs = inputs.to(self.device)
                _, activations = model(inputs, return_activations=True)
                feats = activations['feat']
                feats = F.normalize(feats)
                embedding_list.append(feats.cpu())
                label_list.append(labels.cpu())

        return torch.cat(embedding_list, dim=0), torch.cat(label_list, dim=0)

    def _setup_task_targets(self, dataset):
        if self.current_task == 0:
            return

        print("=" * 50)
        print(f"Task {self.current_task}: Calculating shift targets using MACIL/MSC logic...")

        if self.net_old is None:
            print("Warning: net_old is not available. Cannot calculate shift targets.")
            self.target_prototypes = self.op.clone()
            return

        train_loader = dataset.train_loader

        old_embeds, _ = self._extract_features_with_labels(train_loader, self.net_old)
        new_embeds, _ = self._extract_features_with_labels(train_loader, self.net)

        learned_classes_tensor = torch.tensor(self.learned_classes, device=self.device).long()
        old_prototypes = self.op[learned_classes_tensor]

        gap = self._displacement(old_embeds, new_embeds, old_prototypes)

        self.target_prototypes = self.op.clone()
        self.target_prototypes[learned_classes_tensor] += gap

        self.target_prototypes = F.normalize(self.target_prototypes, p=2, dim=1)

        print("Shift targets calculated and stored for the current task.")
        print("=" * 50)

    def _displacement(self, old_embeds, new_embeds, old_prototypes, sigma=1.0):
        old_embeds = old_embeds.to(self.device)
        new_embeds = new_embeds.to(self.device)
        old_prototypes = old_prototypes.to(self.device)

        DY = new_embeds - old_embeds  # Shape: (N, D)

        dist = torch.cdist(old_prototypes, old_embeds, p=2) ** 2  # Shape: (C, N)

        W = torch.exp(-dist / (2 * sigma ** 2)) + 1e-5  # Shape: (C, N)
        W = W / W.sum(dim=1, keepdim=True)

        displacement = torch.bmm(W.unsqueeze(1), DY.unsqueeze(0).expand(W.size(0), *DY.size()))
        displacement = displacement.squeeze(1)  # Shape: (C, D)

        return displacement

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

            # Regularization loss on Class Prototypes
            if self.current_task > 0 and self.target_prototypes is not None:
                buff_feats = F.normalize(buff_feats)
                dist = 0
                for class_label in torch.unique(buf_labels):
                    if class_label.item() in self.learned_classes:
                        image_class_mask = (buf_labels == class_label)
                        mean_feat = buff_feats[image_class_mask].mean(axis=0)
                        dist += F.mse_loss(mean_feat, self.target_prototypes[class_label])

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
        self.net.eval()

        if self.current_task > 0 and self.target_prototypes is not None:
            print("Updating official prototypes for old classes from shift targets.")
            learned_classes_tensor = torch.tensor(self.learned_classes, device=self.device).long()
            self.op[learned_classes_tensor] = self.target_prototypes[learned_classes_tensor].clone()

        print("Calculating official prototypes for new classes.")
        self.op_sum.zero_()
        self.sample_counts.zero_()
        new_classes_in_this_task = []
        with torch.no_grad():
            for data in dataset.train_loader:
                inputs, labels, _ = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                _, activations = self.net(inputs, return_activations=True)
                feat = F.normalize(activations['feat'])
                for class_label in torch.unique(labels):
                    c = class_label.item()
                    if c not in self.learned_classes:
                        if c not in new_classes_in_this_task:
                            new_classes_in_this_task.append(c)
                        mask = (labels == class_label)
                        self.op_sum[c] += feat[mask].sum(dim=0)
                        self.sample_counts[c] += mask.sum()
        for c in new_classes_in_this_task:
            if self.sample_counts[c] > 0:
                self.op[c] = self.op_sum[c] / self.sample_counts[c]

        self.learned_classes.extend(new_classes_in_this_task)

        print("Saving current model as net_old for the next task.")
        self.net_old = deepcopy(self.net)
        self.net_old.eval()

        self.current_task += 1
        self.eval_prototypes = True
        self.flag = True
        self.get_optimizer()

        self._setup_task_targets(dataset)

        if not self.buffer.is_empty():
            self.net.train()
            buf_inputs, _, _ = self.buffer.get_all_data(transform=self.transform)
            buf_loader = DataLoader(torch.utils.data.TensorDataset(buf_inputs), batch_size=self.args.batch_size,
                                    shuffle=False)
            with torch.no_grad():
                for (data,) in buf_loader:
                    self.net(data)

    def get_optimizer(self):
        self.opt = SGD(self.net.parameters(), lr=self.args.lr)
        if self.args.use_lr_scheduler:
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt, self.args.lr_steps, gamma=0.1)
        else:
            self.scheduler = None
