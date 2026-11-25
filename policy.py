import os; os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT, FAR, FAR_FLOOR
from src.evaluate import tau_at_far, eer

p = argparse.ArgumentParser()
p.add_argument("--metric", default="nfiq2")
p.add_argument("--primary", type=float, default=0.05)                   # attack budget for tuning / headline (1/255)
p.add_argument("--eps", nargs="+", type=float, default=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15])
p.add_argument("--far_budget", type=float, default=(FAR + FAR_FLOOR) / 2) # val FAR budget for tuning (default 0.15%)
p.add_argument("--B", type=int, default=1000)
p.add_argument("--suffix", default="")                                  # appended to output names (ablations)
a = p.parse_args()
FIG, TAB = OUT / "figs", OUT / "tables"; TAB.mkdir(exist_ok=True)
k = lambda e: f"{e:g}"; m = a.metric; PA = f"S_adv_{k(a.primary)}"; tag = m + a.suffix
T = json.load(open(OUT / "tau.json")); tau, tf = T["tau"], T["tau_floor"]
Qd = pd.read_csv(OUT / "quality.csv"); cut = json.load(open(OUT / "quality_cutoffs.json"))[m]
qv, fv, tv = Qd[f"q_{m}"].values, Qd[f"{m}_fail"].values, Qd[f"tier_{m}"].values
ALPHAS = np.round(np.arange(0, 1.0001, 0.005), 3)

def load(split):                                   # all genuine CROSS pairs + all (cross-session) impostors
    P = pd.read_csv(OUT / f"scores_{split}.csv")
    G, I = P[P.kind == "cross"].reset_index(drop=True), P[P.label == 0].reset_index(drop=True)
    A = pd.read_csv(OUT / f"attacks_{split}.csv")
    assert len(A) == (G.S > tf).sum(), "attack set != genuine cross pairs with S > tau_floor"
    G = G.merge(A[["enr", "probe"] + [f"S_adv_{k(e)}" for e in a.eps]], on=["enr", "probe"], how="left")
    for e in a.eps: G[f"S_adv_{k(e)}"] = G[f"S_adv_{k(e)}"].fillna(G.S)   # S <= tau_floor: rejected by every policy anyway
    for D in (G, I): D["q"], D["fail"], D["tier"] = qv[D.probe], fv[D.probe], tv[D.probe]
    return G, I

def thr(D, alpha, qh):                             # tau(Q) = clip(tau - alpha*max(0, Qh - Q), tau_floor, tau); failures keep tau
    t = np.clip(tau - alpha * np.maximum(0, qh - D.q.values), tf, tau)
    return np.where(D.fail.values, tau, t)

def metrics(G, I, tg, ti):
    q1, r = G.tier.values == 1, dict(FAR=(I.S.values > ti).mean() * 100, TAR=(G.S.values > tg).mean() * 100)
    for t in (1, 2, 3): w = G.tier.values == t; r[f"TAR_Q{t}"] = (G.S.values[w] > tg[w]).mean() * 100
    for e in a.eps:
        s = G[f"S_adv_{k(e)}"].values > tg; r[f"TARatt_{k(e)}"] = s.mean() * 100; r[f"TARatt_{k(e)}_Q1"] = s[q1].mean() * 100
    return r

def save(df, name, fmt="%.3f"):
    df.to_csv(TAB / f"{name}.csv", index=False)
    with open(TAB / f"{name}.tex", "w") as f:
        f.write("\\begin{tabular}{" + "l" * df.shape[1] + "}\n\\hline\n" + " & ".join(df.columns).replace("_", "\\_") + " \\\\\n\\hline\n")
        for r in df.itertuples(index=False):
            f.write(" & ".join(fmt % v if isinstance(v, float) else str(v) for v in r).replace("_", "\\_") + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

# ---------- tune alpha, Q_high on VALIDATION only ----------
Gv, Iv = load("val")
QH = dict(c1=cut["c1"], c2=cut["c2"], q90=float(np.quantile(Qd.loc[(Qd.split == "val") & (Qd.session == 2), f"q_{m}"], .9)))
Gr = pd.DataFrame([dict(qh_name=n, qh=qh, alpha=al, **metrics(Gv, Iv, thr(Gv, al, qh), thr(Iv, al, qh)))
                   for n, qh in QH.items() for al in ALPHAS])
obj = f"TARatt_{k(a.primary)}_Q1"
best = Gr[Gr.FAR <= a.far_budget * 100].sort_values([obj, "alpha"], ascending=[False, True]).iloc[0]
al, qh = float(best.alpha), float(best.qh)
tau_g = float(np.clip(tau_at_far(Iv.S.values, best.FAR / 100), tf, tau))           # B2-global: same val FAR, one threshold
Gl = pd.DataFrame([dict(tau_global=t, **metrics(Gv, Iv, np.full(len(Gv), t), np.full(len(Iv), t))) for t in np.linspace(tau, tf, 101)])
Gr.to_csv(TAB / f"policy_grid_val_{tag}.csv", index=False); Gl.to_csv(TAB / f"policy_global_sweep_val_{tag}.csv", index=False)
json.dump(dict(metric=m, alpha=al, qh=qh, qh_name=str(best.qh_name), val_FAR=float(best.FAR), tau_global=tau_g, far_budget=a.far_budget,
               primary=a.primary), open(TAB / f"policy_params_{tag}.json", "w"), indent=1)
print(f"[val tuning] objective = Q1 TAR under attack @ eps={k(a.primary)}/255, s.t. val FAR <= {a.far_budget*100:.3f}%")
print(f"  chosen: Q_high={best.qh_name} ({qh:.3f}), alpha={al:.3f} | val FAR={best.FAR:.3f}% | val Q1 TARatt={best[obj]:.1f}% "
      f"| B2-global tau={tau_g:.4f} (tau={tau:.4f}, tau_floor={tf:.4f})")

# ---------- evaluate on held-out TEST ----------
POL = dict(B1_fixed=lambda D: np.full(len(D), tau), B2_adaptive=lambda D: thr(D, al, qh),
           B2_global=lambda D: np.full(len(D), tau_g), floor_global=lambda D: np.full(len(D), tf))
rows = []
for split in ["val", "test"]:
    G, I = (Gv, Iv) if split == "val" else load("test")
    for name, f in POL.items(): rows.append(dict(split=split, policy=name, **metrics(G, I, f(G), f(I))))
    if split == "test": Gt, It = G, I
R = pd.DataFrame(rows); save(R, f"policy_results_{tag}")
cols = ["policy", "FAR", "TAR", "TAR_Q1", "TAR_Q2", "TAR_Q3", f"TARatt_{k(a.primary)}", f"TARatt_{k(a.primary)}_Q1"]
save(R[R.split == "test"][cols].reset_index(drop=True), f"policy_test_main_{tag}", fmt="%.2f")
print(f"\n[test] cross EER = {eer(np.r_[Gt.S, It.S], np.r_[np.ones(len(Gt)), np.zeros(len(It))])*100:.2f}% | policies (all %):")
print(R[R.split == "test"][cols].round(2).to_string(index=False))

# cluster-bootstrap CIs (by finger) for test differences
acc = {n: (Gt.S.values > f(Gt), Gt[PA].values > f(Gt)) for n, f in POL.items()}
q1 = Gt.tier.values == 1; grp = [ix.to_numpy() for ix in Gt.groupby("finger").groups.values()]; rng = np.random.default_rng(0)
def diffs(ix):
    out = {}
    for A_, B_ in [("B2_adaptive", "B1_fixed"), ("B2_adaptive", "B2_global")]:
        for lab, j, w in [("TAR", 0, None), ("TAR_Q1", 0, q1), ("TARatt", 1, None), ("TARatt_Q1", 1, q1)]:
            s = ix if w is None else ix[w[ix]]
            out[f"{A_}-{B_}|{lab}"] = (acc[A_][j][s].mean() - acc[B_][j][s].mean()) * 100
    return out
pt = diffs(np.arange(len(Gt)))
bs = pd.DataFrame([diffs(np.concatenate([grp[i] for i in rng.integers(len(grp), size=len(grp))])) for _ in range(a.B)])
CI = pd.DataFrame([dict(comparison=c.split("|")[0], metric=c.split("|")[1] + ("" if "att" not in c else f"@{k(a.primary)}"),
                        diff_pp=pt[c], lo=bs[c].quantile(.025), hi=bs[c].quantile(.975),
                        significant=bool(bs[c].quantile(.025) > 0 or bs[c].quantile(.975) < 0)) for c in pt])
save(CI, f"policy_ci_test_{tag}", fmt="%.2f"); print("\nTest differences (percentage points, 95% cluster-bootstrap CI):\n", CI.round(2).to_string(index=False))

# B4: hard rejection curves on test (genuine pairs accepted by B1)
D = Gt[Gt.S > tau]; succ = D[PA].values <= tau; rows = []
for kind, v in [("quality", D.q.values), ("margin", D.S.values - tau)]:
    for r in np.unique(np.r_[0, np.quantile(v, np.linspace(0, 1, 101))]):
        rej = v < r
        rows.append(dict(rule=kind, cutoff=r, genuine_rejected=rej.mean() * 100, attacks_prevented=(succ & rej).sum() / max(succ.sum(), 1) * 100,
                         ASR_remaining=succ[~rej].mean() * 100 if (~rej).any() else np.nan))
B4 = pd.DataFrame(rows); B4.to_csv(TAB / f"b4_test_{tag}.csv", index=False)
print(f"\nB4 (test, eps={k(a.primary)}/255): attacks prevented at a given genuine-rejection rate")
for kind in ["quality", "margin"]:
    b = B4[B4.rule == kind]
    print(f"  {kind:8s}", " | ".join(f"{g}% rejected -> {np.interp(g, b.genuine_rejected, b.attacks_prevented):.1f}% prevented" for g in (10, 20, 30)),
          "  (random rejection = same % prevented)")

# ---------- figures ----------
qs = np.linspace(0, 1, 201); fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(qs, np.clip(tau - al * np.maximum(0, qh - qs), tf, tau), "b-", label=f"B2: α={al:.3f}, Q_high={qh:.2f}")
for y, l, c in [(tau, "τ_fixed (FAR 0.1%)", "k"), (tf, "τ_floor (FAR 0.2%)", "r"), (tau_g, "B2-global", "g")]: ax.axhline(y, ls="--", c=c, lw=1, label=l)
for c_ in (cut["c1"], cut["c2"]): ax.axvline(c_, c="gray", lw=.8, ls=":")
ax.set_xlabel(f"probe quality Q ({m.upper()})"); ax.set_ylabel("threshold"); ax.legend(fontsize=8); fig.savefig(FIG / f"tau_of_Q_{tag}.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for n in QH: g = Gr[Gr.qh_name == n]; ax.plot(g.FAR, g[obj], "-", label=f"adaptive, Q_high={n}")
ax.plot(Gl.FAR, Gl[obj], "k--", label="global threshold"); ax.plot(best.FAR, best[obj], "r*", ms=14, label="chosen B2")
ax.axvline(a.far_budget * 100, c="gray", ls=":", label="FAR budget")
ax.set_xlabel("val FAR (%)"); ax.set_ylabel(f"val Q1 TAR under attack @ε={k(a.primary)}/255 (%)"); ax.legend(fontsize=8)
ax.set_title("Validation frontier: quality-adaptive vs global threshold"); fig.savefig(FIG / f"policy_frontier_val_{tag}.png", dpi=150, bbox_inches="tight")

fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True); rt = R[R.split == "test"].set_index("policy")
for ax, suf, ttl in [(axs[0], "", "all genuine cross pairs"), (axs[1], "_Q1", "Q1 (low-quality) probes")]:
    for n in POL: ax.plot(a.eps, [rt.loc[n, f"TARatt_{k(e)}{suf}"] for e in a.eps], "o-", label=n)
    ax.set_xscale("log"); ax.set_xlabel("ε (×1/255)"); ax.set_title(f"Test TAR under attack: {ttl}")
axs[0].set_ylabel("TAR (%)"); axs[0].legend(); fig.savefig(FIG / f"policy_tar_vs_eps_test_{tag}.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for kind, c in [("quality", "tab:blue"), ("margin", "tab:purple")]: b = B4[B4.rule == kind]; ax.plot(b.genuine_rejected, b.attacks_prevented, c=c, label=f"{kind} rejection")
ax.plot([0, 100], [0, 100], "k:", label="random rejection"); ax.set_xlabel("genuine users rejected (%)"); ax.set_ylabel("successful attacks prevented (%)")
ax.set_title(f"B4 (test, ε={k(a.primary)}/255)"); ax.legend(); fig.savefig(FIG / f"b4_rejection_test_{tag}.png", dpi=150, bbox_inches="tight")
print(f"\nSaved tables -> {TAB}  figures -> {FIG}")
