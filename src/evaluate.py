import numpy as np, torch
from sklearn.metrics import roc_curve
from config import BATCH
from src.dataset import Imgs

@torch.no_grad()
def embed_ds(model, ds, device, bs=64):
    model.eval()
    out = []
    for i in range(0, len(ds), bs):
        xb = torch.stack([ds[j] for j in range(i, min(i + bs, len(ds)))]).to(device)
        out.append(model(xb).cpu())
    return torch.cat(out).numpy()

def full_array(E, rows, n):
    full = np.full((n, E.shape[1]), np.nan, dtype=np.float32)
    full[rows] = E
    return full                                    # row index of images.csv -> embedding

def embed_rows(model, df, rows, device):           # loads from disk each call; fine for one-off use
    return full_array(embed_ds(model, Imgs(df, rows), device, BATCH * 2), rows, len(df))

def pair_scores(E, pairs):
    return (E[pairs.enr.values] * E[pairs.probe.values]).sum(1)

def eer(scores, labels):
    far, tpr, _ = roc_curve(labels, scores)
    frr = 1 - tpr
    k = np.nanargmin(np.abs(far - frr))
    return float((far[k] + frr[k]) / 2)

def tau_at_far(imp_scores, far):
    return float(np.quantile(imp_scores, 1 - far, method="higher"))   # conservative: realized FAR <= far; accept if S > tau

def tar_at_far(scores, labels, far):
    tau = tau_at_far(scores[labels == 0], far)
    return float((scores[labels == 1] > tau).mean()), tau

def boot_ci(P, fn, B=1000, seed=0):
    """Cluster bootstrap by finger. P needs columns S, label, finger. fn(scores, labels) -> float."""
    rng = np.random.default_rng(seed)
    groups = {f: g.index.to_numpy() for f, g in P.groupby("finger")}
    fingers = np.array(list(groups))
    S, y, vals = P.S.values, P.label.values, []
    for _ in range(B):
        idx = np.concatenate([groups[f] for f in rng.choice(fingers, len(fingers), replace=True)])
        vals.append(fn(S[idx], y[idx]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]