import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from copy import deepcopy

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

def compute_drs_projection_from_features(net_old, dataloader, device, topk=64):
    print("--- DRS Computation for 'linear' Layer ONLY ---")

    model_for_feature_extraction = deepcopy(net_old).to(device).eval()

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

    MAX_BATCHES_FOR_FEATURES = 100

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