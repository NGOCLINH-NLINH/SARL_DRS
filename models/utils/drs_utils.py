import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy


def compute_feature_subspaces(net_old, dataloader_new, target_layers, device, variance_threshold=1.0):
    print("\n--- Computing Feature Subspaces for Regularization (Explained Variance Threshold: {variance_threshold*100}%)---")

    model_for_feature_extraction = deepcopy(net_old).to(device).eval()

    print(f"Targeting {len(target_layers)} layers for feature space construction: {target_layers}")

    feature_accumulator = {name: [] for name in target_layers}
    hooks = []

    def get_output_features_hook(name):
        def hook(module, input, output):
            flat_output = output.detach().view(output.shape[0], -1).cpu()
            feature_accumulator[name].append(flat_output)

        return hook

    for name, module in model_for_feature_extraction.named_modules():
        if name in target_layers:
            hooks.append(module.register_forward_hook(get_output_features_hook(name)))

    # MAX_BATCHES_FOR_FEATURES = 319

    with torch.no_grad():
        for i, (inputs, _, _) in enumerate(tqdm(dataloader_new, desc="Collecting Output Features")):
            # if i >= MAX_BATCHES_FOR_FEATURES:
            #     break
            model_for_feature_extraction(inputs.to(device))

    for hook in hooks:
        hook.remove()

    subspaces = {}
    print("Computing SVD to define subspaces...")

    for name, feat_list in tqdm(feature_accumulator.items(), desc="Computing SVD"):
        if not feat_list:
            continue
        try:
            # all_feats = torch.cat(feat_list, dim=0).to(device)
            # U, S, Vh = torch.linalg.svd(all_feats, full_matrices=False)
            #
            # k = min(topk, Vh.shape[0])
            #
            # basis_vectors = Vh[:k, :]
            # subspaces[name] = basis_vectors
            X = torch.cat(feat_list, dim=0).to(device)
            U_small, S_small, Vh_small = torch.linalg.svd(X @ X.T, full_matrices=False)
            explained_variance = S_small / S_small.sum()

            cumulative_variance = torch.cumsum(explained_variance, dim=0)
            k_tensor = torch.where(cumulative_variance >= variance_threshold)[0]

            if len(k_tensor) == 0:
                k = len(S_small)
            else:
                k = k_tensor[0].item() + 1

            print(
                f"  - Layer '{name}': k = {k} components selected to capture {variance_threshold * 100}% of variance.")

            S_inv_sqrt = torch.diag(1.0 / torch.sqrt(S_small + 1e-8))
            basis_vectors = S_inv_sqrt @ U_small.T @ X

            subspaces[name] = basis_vectors[:k, :]

        except Exception as e:
            print(f"Could not compute SVD for layer {name} due to error: {e}. Skipping.")
            continue

    del model_for_feature_extraction
    torch.cuda.empty_cache()

    print("--- Feature Subspace Computation Finished ---")
    return subspaces

def compute_drs_projection(model, dataloader, device, topk=64):
    model.eval()
    grads_by_param = {name: [] for name, p in model.named_parameters() if p.requires_grad}
    print("Collecting gradients for DRS projection...")

    for inputs, labels, _ in tqdm(dataloader, desc="Collecting Gradients"):
        inputs, labels = inputs.to(device), labels.to(device)

        model.zero_grad()
        outputs, _ = model(inputs, return_activations=True)
        loss = F.cross_entropy(outputs, labels)

        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                grads_by_param[name].append(p.grad.view(-1).detach().cpu())

    model.zero_grad()
    projections = {}
    print("\nComputing SVD for projection matrices...")

    for name, grads_list in tqdm(grads_by_param.items(), desc="Computing SVD"):
        if not grads_list:
            continue

        G = torch.stack(grads_list, dim=0)
        G = G - G.mean(dim=0, keepdim=True)
        cov = G.T @ G

        # try:
        #     U, S, Vh = np.linalg.svd(cov.numpy(), full_matrices=False)
        #     P = torch.from_numpy(U[:, :topk]).float().to(device)
        #     projections[name] = P
        # except np.linalg.LinAlgError as e:
        #     print(f"SVD failed for parameter {name}: {e}")
        #     continue

        try:
            U_small, S_small, Vh_small = torch.linalg.svd(G @ G.T, full_matrices=False)

            S_inv = torch.diag(1.0 / torch.sqrt(S_small + 1e-8))
            P = G.T @ U_small @ S_inv

            k = min(topk, P.shape[1])
            projections[name] = P[:, :k]

        except torch.linalg.LinAlgError as e:
            print(f"SVD failed for parameter {name}: {e}. Skipping projection for this parameter.")
            continue

    return projections


def create_subtracted_model(net_initial, net_old):
    net_tilde = deepcopy(net_initial)

    params_tilde = net_tilde.parameters()
    params_initial = net_initial.parameters()
    params_old = net_old.parameters()

    for p_tilde, p_initial, p_old in zip(params_tilde, params_initial, params_old):
        p_tilde.data = 2 * p_initial.data - p_old.data

    return net_tilde

def compute_drs_projection_from_features(net_initial, net_old, dataloader, device, topk=64):
    print("--- DRS Computation for 'linear' Layer ONLY ---")

    # model_for_feature_extraction = deepcopy(net_old).to(device).eval()

    print("Creating subtracted model (W_tilde = 2*W_0 - W_{t-1})...")
    model_for_feature_extraction = create_subtracted_model(net_initial, net_old).to(device).eval()

    TARGET_MODULE_NAME = 'linear'
    print(f"Collecting input features from the target module: '{TARGET_MODULE_NAME}'")

    feature_accumulator = []
    hooks = []

    def get_features_hook(model, input):
        feature_accumulator.append(input[0].detach().cpu())

    target_module_found = False
    for name, module in model_for_feature_extraction.named_modules():
        if name == TARGET_MODULE_NAME:
            hooks.append(module.register_forward_pre_hook(get_features_hook))
            target_module_found = True
            break

    if not target_module_found:
        print(f"Error: Module '{TARGET_MODULE_NAME}' not found in the model. Aborting DRS.")
        return {}

    MAX_BATCHES_FOR_FEATURES = 312

    with torch.no_grad():
        for i, (inputs, _, _) in enumerate(tqdm(dataloader, desc=f"Feature Collection for '{TARGET_MODULE_NAME}'")):
            if i >= MAX_BATCHES_FOR_FEATURES:
                break
            model_for_feature_extraction(inputs.to(device))

    for hook in hooks:
        hook.remove()

    projections = {}
    print(f"Computing SVD for '{TARGET_MODULE_NAME}'...")

    if feature_accumulator:
        try:
            all_feats = torch.cat(feature_accumulator, dim=0)  # Shape: [N_total, D_in]
            n_samples = all_feats.shape[0]

            flat_feats_gpu = all_feats.to(device)

            cov = flat_feats_gpu.T @ flat_feats_gpu / n_samples

            U, S, Vh = torch.linalg.svd(cov, full_matrices=False)

            k = min(topk, U.shape[1])

            projections[TARGET_MODULE_NAME] = U[:, :k]

        except Exception as e:
            print(f"Could not compute SVD for layer {TARGET_MODULE_NAME} due to error: {e}")

    del model_for_feature_extraction
    torch.cuda.empty_cache()

    print("--- DRS Computation Finished ---")
    return projections