import torch
import torch.nn.functional as F
from tqdm import tqdm
from copy import deepcopy

def compute_drs_projection_from_gradient(net_old, dataloader_old, device, topk=64):
    print("\n--- DRS Computation from GRADIENTS ---")

    MAX_BATCHES_FOR_DRS = 50

    DRS_TARGET_PARAMS = ['linear.weight']

    USE_FLOAT16 = True

    grad_dtype = torch.float16 if USE_FLOAT16 else torch.float32
    model = deepcopy(net_old).to(device).eval()

    grads_by_param = {name: [] for name in DRS_TARGET_PARAMS}
    print(f"Collecting gradients for {len(DRS_TARGET_PARAMS)} parameters: {DRS_TARGET_PARAMS}")
    print(f"Using a maximum of {MAX_BATCHES_FOR_DRS} batches from the old task data (buffer).")

    for i, (inputs, labels, _) in enumerate(tqdm(dataloader_old, desc="Collecting Gradients")):
        if i >= MAX_BATCHES_FOR_DRS:
            break

        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()

        outputs, _ = model(inputs, return_activations=True)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()

        for name, p in model.named_parameters():
            if name in DRS_TARGET_PARAMS and p.grad is not None:
                grads_by_param[name].append(p.grad.view(-1).detach().to(device='cpu', dtype=grad_dtype))

    model.zero_grad()
    projections = {}
    print("\nComputing SVD on collected gradients...")

    for name, grads_list in tqdm(grads_by_param.items(), desc="Computing SVD"):
        if not grads_list:
            continue

        try:
            G = torch.stack(grads_list, dim=0).to(device=device, dtype=torch.float32)
            G = G - G.mean(dim=0, keepdim=True)

            U_small, S_small, Vh_small = torch.linalg.svd(G @ G.T, full_matrices=False)
            S_inv = torch.diag(1.0 / (torch.sqrt(S_small) + 1e-8))
            P = G.T @ U_small @ S_inv

            k = min(topk, P.shape[1])
            projections[name] = P[:, :k]

        except Exception as e:
            print(f"SVD failed for parameter {name}: {e}. Skipping.")
            continue

    del model
    torch.cuda.empty_cache()

    print("--- DRS Computation from Gradients Finished ---")
    return projections