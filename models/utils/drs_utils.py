import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

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