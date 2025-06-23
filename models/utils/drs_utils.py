import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

def compute_drs_projection(model, dataloader, device, topk=64):
    model.eval()
    features_by_param = {}

    with torch.no_grad():
        for inputs, labels, _ in tqdm(dataloader, desc="Collecting features for DRS"):
            inputs = inputs.to(device)
            _, activations = model(inputs, return_activations=True)
            if 'feat' not in activations:
                continue
            feat = activations['feat']
            feat = F.normalize(feat, dim=-1).detach().cpu()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in features_by_param:
                        features_by_param[name] = []
                    features_by_param[name].append(feat)

    projections = {}
    for name, feats in features_by_param.items():
        X = torch.cat(feats, dim=0)  # [N, D]
        X = X - X.mean(dim=0, keepdim=True)
        X_np = X.numpy()
        cov = X_np.T @ X_np  # D x D
        U, S, Vh = np.linalg.svd(cov)
        P = torch.from_numpy(U[:, :topk]).float().to(device)  # D x topk
        projections[name] = P

    return projections