import argparse, json, time, numpy as np, pandas as pd, torch
from config import OUT
from src.dataset import Imgs
from src.model import load_model

p = argparse.ArgumentParser()
p.add_argument("--splits", nargs="+", default=["test", "val"])
p.add_argument("--kinds", nargs="+", default=["cross"])
p.add_argument("--eps", nargs="+", type=float,                            # in units of 1/255: fine grid + proposal grid
               default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 1, 2, 4, 8])
p.add_argument("--steps", type=int, default=50)
p.add_argument("--restarts", type=int, default=1)
p.add_argument("--n_noise", type=int, default=10)                         # B3: worst of n random sign-noise draws
p.add_argument("--bisect", type=int, default=10)                          # eps* bisection steps (0 = off)
p.add_argument("--eps_hi", type=float, default=0.5)                       # eps* search range [0, eps_hi]/255
p.add_argument("--bs", type=int, default=128)
p.add_argument("--limit", type=int, default=0)                            # smoke test: first N pairs per split
p.add_argument("--seed", type=int, default=0)
p.add_argument("--sample", type=int, default=0)                           # random subset of ACCEPTED pairs (strength check)
p.add_argument("--suffix", default="")                                    # appended to the output file name
a = p.parse_args()
torch.manual_seed(a.seed); torch.backends.cudnn.benchmark = True
dev = "cuda"
T = json.load(open(OUT / "tau.json")); tau, tau_floor = T["tau"], T["tau_floor"]
model = load_model(OUT / "ckpt" / f"{T['ckpt']}.pt", dev)              # eval mode: BN/dropout frozen
for q in model.parameters(): q.requires_grad_(False)
df = pd.read_csv(OUT / "images.csv"); Q = pd.read_csv(OUT / "quality.csv")
QCOLS = ["q_nfiq2", "tier_nfiq2", "nfiq2_fail", "q_mclfiq", "tier_mclfiq", "mclfiq_fail"]

def sim(x, ee): return (model(x) * ee).sum(1)

def pgd(xp, ee, eps):
    """L_inf PGD dodging (Madry et al.): minimise S(x_p + delta, x_e), ||delta||_inf <= eps, x in [0,1].
    eps: scalar or per-pair tensor (pixel units). Step 2.5*eps/steps. Returns lowest S over iterates and restarts."""
    E = torch.as_tensor(eps, device=dev, dtype=xp.dtype).reshape(-1, 1, 1, 1); alpha = 2.5 * E / a.steps
    best = torch.full((len(xp),), np.inf, device=dev)
    for _ in range(a.restarts):
        x = (xp + (torch.rand_like(xp) * 2 - 1) * E).clamp(0, 1)
        for _ in range(a.steps):
            x.requires_grad_(True)
            s = sim(x, ee)
            g, = torch.autograd.grad(s.sum(), x)
            best = torch.minimum(best, s.detach())
            x = torch.min(torch.max(x.detach() - alpha * g.sign(), xp - E), xp + E).clamp(0, 1)
        with torch.no_grad(): best = torch.minimum(best, sim(x, ee))
        assert ((x - xp).abs().amax(dim=(1, 2, 3)) <= E.flatten() + 1e-6).all()
    return best

@torch.no_grad()
def noise(xp, ee, eps):                                                   # B3: equal-budget random control
    best = torch.full((len(xp),), np.inf, device=dev)
    for _ in range(a.n_noise):
        x = (xp + eps * (torch.randint(0, 2, xp.shape, device=dev) * 2 - 1)).clamp(0, 1)
        best = torch.minimum(best, sim(x, ee))
    return best

def eps_star(xp, ee):
    """Per-pair robustness radius: smallest eps (pixel units) at which PGD pushes S <= tau (bisection)."""
    lo = torch.zeros(len(xp), device=dev); hi = torch.full((len(xp),), a.eps_hi / 255, device=dev)
    broken = pgd(xp, ee, hi) <= tau
    for _ in range(a.bisect):
        mid = (lo + hi) / 2; ok = pgd(xp, ee, mid) <= tau
        hi = torch.where(ok, mid, hi); lo = torch.where(ok, lo, mid)
    return torch.where(broken, hi, torch.full_like(hi, np.inf))

for split in a.splits:
    P = pd.read_csv(OUT / f"scores_{split}.csv")
    A = P[(P.label == 1) & P.kind.isin(a.kinds) & (P.S > tau_floor)].reset_index(drop=True)   # covers tau and tau_floor
    if a.limit: A = A.head(a.limit)
    if a.sample: A = A[A.S > tau].sample(min(a.sample, int((A.S > tau).sum())), random_state=a.seed).reset_index(drop=True)
    rows = sorted(set(A.enr) | set(A.probe)); pos = {r: i for i, r in enumerate(rows)}
    X = Imgs(df, rows).x                                                  # uint8, the exact pixels the matcher saw
    ie, ip = A.enr.map(pos).values, A.probe.map(pos).values
    out = {f"{k}_{e:g}": np.zeros(len(A)) for e in a.eps for k in ["S_adv", "S_noise"]}
    S0, ES = np.zeros(len(A)), np.full(len(A), np.nan); t0 = time.time()
    for b in range(0, len(A), a.bs):
        sl = slice(b, b + a.bs)
        xp = X[ip[sl]].to(dev).float() / 255; xe = X[ie[sl]].to(dev).float() / 255
        with torch.no_grad(): ee = model(xe); S0[sl] = sim(xp, ee).cpu().numpy()
        for e in a.eps:
            out[f"S_adv_{e:g}"][sl] = pgd(xp, ee, e / 255).cpu().numpy()
            out[f"S_noise_{e:g}"][sl] = noise(xp, ee, e / 255).cpu().numpy()
        if a.bisect: ES[sl] = eps_star(xp, ee).cpu().numpy() * 255
        print(f"[{split}] {min(b + a.bs, len(A))}/{len(A)} pairs  {time.time() - t0:.0f}s", flush=True)
    dS = np.abs(S0 - A.S.values).max(); print(f"[{split}] max |S_clean - calibrate S| = {dS:.2e}", flush=True)
    assert dS < 1e-2, "clean scores differ from calibrate.py (more than GPU numerical noise)"

    A = A.rename(columns={"S": "S_clean"}).drop(columns=["accepted", "label"])
    A["M"] = A.S_clean - tau; A["acc"] = A.S_clean > tau
    A["eps_star"] = np.where(A.acc, ES, np.nan)                           # in 1/255 units; only defined for accepted pairs
    A = A.join(Q.loc[A.probe, QCOLS].reset_index(drop=True))
    for e in a.eps:
        A[f"S_adv_{e:g}"], A[f"S_noise_{e:g}"] = out[f"S_adv_{e:g}"], out[f"S_noise_{e:g}"]
        A[f"succ_{e:g}"], A[f"succn_{e:g}"] = A[f"S_adv_{e:g}"] <= tau, A[f"S_noise_{e:g}"] <= tau
    A.to_csv(OUT / (f"attacks_{split}" + (f"_limit{a.limit}" if a.limit else "") + a.suffix + ".csv"), index=False)

    acc = A[A.acc]
    print(f"\n[{split}] attacked {len(A)} pairs (S > tau_floor) | accepted at tau: {len(acc)} from {acc.finger.nunique()} fingers")
    for e in a.eps:
        bt = (acc.groupby("tier_nfiq2")[f"succ_{e:g}"].mean() * 100).round(1).to_dict()
        print(f"  eps={e:g}/255  ASR PGD={acc[f'succ_{e:g}'].mean()*100:5.1f}%  noise={acc[f'succn_{e:g}'].mean()*100:5.1f}%  "
              f"mean S drop={(acc.S_clean - acc[f'S_adv_{e:g}']).mean():.3f} | PGD ASR by NFIQ2 tier {bt}")
    if a.bisect:
        print(f"  eps*: median {acc.eps_star.median():.4f}/255  IQR {acc.eps_star.quantile(.25):.4f}-{acc.eps_star.quantile(.75):.4f} | "
              f"not broken at {a.eps_hi}/255: {np.isinf(acc.eps_star).sum()} | Spearman(eps*, M) = {acc.eps_star.corr(acc.M, method='spearman'):.3f}")
        print("  median eps* by NFIQ2 tier:", acc.groupby("tier_nfiq2").eps_star.median().round(4).to_dict())
