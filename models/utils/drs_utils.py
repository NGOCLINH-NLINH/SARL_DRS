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
            model.zero_grad()
            output, activations = model(inputs, return_activations=True)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name not in features_by_param:
                        features_by_param[name] = []
                    if "feat" in activations:
                        feat = activations['feat']
                    else:
                        feat = output
                    feat = F.normalize(feat, dim=-1).detach().cpu()
                    features_by_param[name].append(feat)
    projections = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            projections.append(None)
            continue

        feats = features_by_param.get(name, None)
        if feats is None or len(feats) == 0:
            projections.append(None)
            continue

        X = torch.cat(feats, dim=0)
        X = X - X.mean(dim=0, keepdim=True)
        X_np = X.numpy()
        U, S, Vh = np.linalg.svd(X_np.T @ X_np)
        P = torch.from_numpy(U[:, :topk]).float().to(device)
        projections.append(P)

    return projections