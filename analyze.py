import os; os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse, warnings, numpy as np, pandas as pd, statsmodels.api as sm
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT

p = argparse.ArgumentParser()
p.add_argument("--file", default="attacks_test")                         # attacks_<split>[_limitN]
p.add_argument("--eps", nargs="+", type=float, default=[0.02, 0.03, 0.05, 0.075])   # unsaturated budgets (1/255)
p.add_argument("--primary", type=float, default=0.05)
p.add_argument("--dm", type=float, default=0.05)                         # margin band width
p.add_argument("--nb", type=int, default=4)                              # number of bands (last one open-ended)
p.add_argument("--B", type=int, default=1000)                            # cluster-bootstrap resamples
a = p.parse_args()
warnings.filterwarnings("ignore"); pd.set_option("display.width", 200, "display.max_columns", 20)
FIG, TAB = OUT / "figs", OUT / "tables"; TAB.mkdir(exist_ok=True)
tag = a.file.replace("attacks_", "")
A = pd.read_csv(OUT / f"{a.file}.csv")
k = lambda e: f"{e:g}"
ALL_EPS = sorted(float(c[6:]) for c in A.columns if c.startswith("S_adv_"))
BANDS = [f"[{i*a.dm:.2f},{(i+1)*a.dm:.2f})" for i in range(a.nb - 1)] + [f">={(a.nb-1)*a.dm:.2f}"]
CTRLS = ["none", "M", "bandM", "logM"]

def prep(metric, keep_fail=False):                                       # accepted pairs; Q = probe quality
    d = A[A.acc & (keep_fail | ~A[f"{metric}_fail"])].copy()
    d["Q10"], d["tier"] = d[f"q_{metric}"] * 10, d[f"tier_{metric}"]    # Q10: coefficients are per +0.1 Q
    d["logM"], d["logE"] = np.log(d.M), np.log(d.eps_star)
    d["band"] = np.minimum((d.M / a.dm).astype(int), a.nb - 1)
    assert np.isfinite(d.logE).all(), "eps* not finite for some accepted pairs"
    return d.reset_index(drop=True)

def boot(d, seed=0):                                                     # finger-clustered resamples (reused by all stats)
    rng = np.random.default_rng(seed)
    g = {f: ix.to_numpy() for f, ix in d.groupby("finger").groups.items()}; F = np.array(list(g))
    return [d.iloc[np.concatenate([g[f] for f in rng.choice(F, len(F))])] for _ in range(a.B)]

def cival(DB, fn):
    v = np.array([fn(x) for x in DB], float)
    return tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).any() else (np.nan, np.nan)

def design(d, ctrl):
    cols = [np.ones(len(d)), d.Q10.values]
    if ctrl == "M": cols.append(d.M.values)
    if ctrl == "logM": cols.append(d.logM.values)
    if ctrl == "bandM": cols += [(d.band.values == b).astype(float) for b in range(1, a.nb) if (d.band.values == b).any()]
    return np.column_stack(cols)

def firth(y, X, maxit=100, tol=1e-8):
    """Firth (1993) bias-reduced logistic regression: finite estimates even under (quasi-)separation."""
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        pi = 1 / (1 + np.exp(-np.clip(X @ b, -30, 30))); W = pi * (1 - pi)
        inv = np.linalg.inv(X.T @ (X * W[:, None]))
        h = W * np.einsum("ij,jk,ik->i", X, inv, X)                       # hat-matrix diagonal
        step = inv @ (X.T @ (y - pi + h * (0.5 - pi)))
        step *= min(1.0, 5 / np.abs(step).max()); b += step              # cap step size for stability
        if np.abs(step).max() < tol: break
    return b

def coef(d, ctrl, y):                                                    # coefficient of Q10
    try:
        Z, yy = design(d, ctrl), d[y].values.astype(float)
        if y == "logE": return sm.OLS(yy, Z).fit().params[1]
        if yy.min() == yy.max(): return np.nan                           # outcome constant in this resample
        return firth(yy, Z)[1]
    except Exception: return np.nan

def save(df, name, fmt="%.3f"):
    df.to_csv(TAB / f"{name}_{tag}.csv", index=False)
    with open(TAB / f"{name}_{tag}.tex", "w") as f:
        f.write("\\begin{tabular}{" + "l" * df.shape[1] + "}\n\\hline\n" + " & ".join(df.columns).replace("_", "\\_") + " \\\\\n\\hline\n")
        for r in df.itertuples(index=False):
            f.write(" & ".join(fmt % v if isinstance(v, float) else str(v) for v in r).replace("_", "\\_") + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

R = {}
for metric, keep in [("nfiq2", False), ("mclfiq", False), ("nfiq2", True)]:
    d = prep(metric, keep); R[metric + ("+fail" if keep else "")] = (d, boot(d))
d, DB = R["nfiq2"]
P = f"succ_{k(a.primary)}"
print(f"[{tag}] accepted pairs: {len(A[A.acc])} | NFIQ2 analysis set: {len(d)} pairs, {d.finger.nunique()} fingers | B={a.B}")

# 1. Confounder in plain view: margin and robustness by quality tier
rows = []
for name in ["nfiq2", "mclfiq"]:
    dd, DD = R[name]
    for t in (1, 2, 3):
        s = dd[dd.tier == t]; lo, hi = cival(DD, lambda x: x.loc[x.tier == t, P].mean())
        rows.append(dict(metric=name, tier=f"Q{t}", n=len(s), fingers=s.finger.nunique(), mean_M=s.M.mean(),
                         median_eps_star=s.eps_star.median(), ASR=s[P].mean() * 100, lo=lo * 100, hi=hi * 100))
T1 = pd.DataFrame(rows); save(T1, "tier_summary"); print(f"\nTier summary (ASR at eps={k(a.primary)}/255):\n", T1.round(3).to_string(index=False))

# 2. Margin-band stratification: ASR by tier within narrow margin bands
rows = []
for e in a.eps:
    col = f"succ_{k(e)}"
    for b in range(a.nb):
        for t in (1, 2, 3):
            s = d[(d.band == b) & (d.tier == t)]
            lo, hi = cival(DB, lambda x: x.loc[(x.band == b) & (x.tier == t), col].mean())
            rows.append(dict(eps=e, band=BANDS[b], tier=f"Q{t}", n=len(s), fingers=s.finger.nunique(),
                             ASR=s[col].mean() * 100 if len(s) else np.nan, lo=lo * 100, hi=hi * 100))
T2 = pd.DataFrame(rows); save(T2, "bands")
print(f"\nASR (%) by margin band x tier at eps={k(a.primary)}/255 (n in brackets):")
t2 = T2[T2.eps == a.primary]
print(t2.assign(cell=t2.ASR.round(1).astype(str) + " [" + t2.n.astype(str) + "]").pivot(index="band", columns="tier", values="cell").reindex(BANDS))

# 3. Regression: Q effect before vs after margin control (H0: beta_Q = 0)
rows = []
for name, (dd, DD) in R.items():
    outs = [f"succ_{k(e)}" for e in a.eps] + ["logE"]
    if name.endswith("+fail"): outs = [P, "logE"]
    for y in outs:
        for c in CTRLS:
            b = coef(dd, c, y); bs = np.array([coef(x, c, y) for x in DD]); lo, hi = np.nanpercentile(bs, [2.5, 97.5])
            rows.append(dict(metric=name, outcome="log_eps*" if y == "logE" else f"success@{y[5:]}", control=c, n=len(dd),
                             beta=b, ratio=np.exp(b), lo=np.exp(lo), hi=np.exp(hi), boot_ok=int(np.isfinite(bs).sum()),
                             reject_H0=bool(lo > 0 or hi < 0)))
    print(f"  regressions done: {name}", flush=True)
T3 = pd.DataFrame(rows); save(T3, "regression")
print("\nQ effect per +0.1 Q (ratio = odds ratio for success; = multiplicative change in eps* for log_eps*):")
print(T3[T3.outcome.isin([f"success@{k(a.primary)}", "log_eps*"])].round(3).to_string(index=False))

# 4. B3: PGD vs equal-budget random noise (all accepted pairs)
da, DA = R["nfiq2+fail"]; rows = []
for e in ALL_EPS:
    c1, c2 = f"succ_{k(e)}", f"succn_{k(e)}"; l1, h1 = cival(DA, lambda x: x[c1].mean()); l2, h2 = cival(DA, lambda x: x[c2].mean())
    rows.append(dict(eps=e, ASR_PGD=da[c1].mean() * 100, lo=l1 * 100, hi=h1 * 100, ASR_noise=da[c2].mean() * 100, lo_n=l2 * 100, hi_n=h2 * 100))
T4 = pd.DataFrame(rows); save(T4, "b3"); print("\nB3 PGD vs random noise:\n", T4.assign(eps=T4.eps.map(k)).round(1).to_string(index=False))

# 5. Finger fixed effects: is the effect within fingers (image quality) or between fingers (finger identity)?
import statsmodels.formula.api as smf
rows = []
for name in ["nfiq2", "mclfiq"]:
    dd = R[name][0]; dw = dd[dd.groupby("finger").Q10.transform("std") > 0]
    for lab, fml in [("logM", "logE ~ Q10 + logM"), ("logM+finger_FE", "logE ~ Q10 + logM + C(finger)")]:
        r = smf.ols(fml, dw).fit(cov_type="cluster", cov_kwds={"groups": dw.finger}); lo, hi = r.conf_int().loc["Q10"]
        rows.append(dict(metric=name, model=lab, n=len(dw), fingers=dw.finger.nunique(), ratio=np.exp(r.params["Q10"]), lo=np.exp(lo),
                         hi=np.exp(hi), p=r.pvalues["Q10"], within_finger_sd_Q=dd.groupby("finger")[f"q_{name}"].std().median()))
T5 = pd.DataFrame(rows); save(T5, "finger_fe")
print("\nWithin-finger check (eps* ratio per +0.1 Q, finger-clustered SE):\n", T5.round(4).to_string(index=False))

# Figures
C = {1: "tab:red", 2: "tab:orange", 3: "tab:green"}
fig, ax = plt.subplots(figsize=(6.5, 4))
for t in (1, 2, 3):
    s = d[d.tier == t]; ax.plot(ALL_EPS, [s[f"succ_{k(e)}"].mean() * 100 for e in ALL_EPS], "o-", c=C[t], label=f"Q{t} (n={len(s)})")
ax.plot(ALL_EPS, [da[f"succn_{k(e)}"].mean() * 100 for e in ALL_EPS], "k--", label="random noise (B3)")
ax.set_xscale("log"); ax.set_xlabel("ε (×1/255)"); ax.set_ylabel("ASR (%)"); ax.set_title("ASR vs budget by NFIQ2 tier (unadjusted)"); ax.legend()
fig.savefig(FIG / f"asr_vs_eps_{tag}.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(7, 4)); w = 0.27
for i, t in enumerate((1, 2, 3)):
    s = t2[t2.tier == f"Q{t}"].set_index("band").reindex(BANDS); x = np.arange(a.nb) + (i - 1) * w
    ax.bar(x, s.ASR, w, color=C[t], label=f"Q{t}", yerr=[(s.ASR - s.lo).clip(0), (s.hi - s.ASR).clip(0)], capsize=3)
ax.set_xticks(np.arange(a.nb), BANDS); ax.set_xlabel("clean margin band M"); ax.set_ylabel("ASR (%)")
ax.set_title(f"ASR within margin bands, ε={k(a.primary)}/255 (95% cluster-bootstrap CI)"); ax.legend()
fig.savefig(FIG / f"asr_bands_{tag}.png", dpi=150, bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 4.5))
for t in (1, 2, 3):
    s = d[d.tier == t]; ax.scatter(s.M, s.eps_star, s=9, alpha=.6, c=C[t], label=f"Q{t}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("clean margin M = S − τ"); ax.set_ylabel("ε* (×1/255)")
ax.set_title(f"Robustness radius vs margin (Spearman {d.eps_star.corr(d.M, method='spearman'):.2f})"); ax.legend()
fig.savefig(FIG / f"epsstar_vs_M_{tag}.png", dpi=150, bbox_inches="tight")

F = T3[T3.metric.isin(["nfiq2", "mclfiq"]) & T3.outcome.isin([f"success@{k(a.primary)}", "log_eps*"])].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7, 0.32 * len(F) + 1)); y = np.arange(len(F))[::-1]
ax.errorbar(F.ratio, y, xerr=[(F.ratio - F.lo).clip(0), (F.hi - F.ratio).clip(0)], fmt="o", c="k", capsize=3)
ax.axvline(1, c="gray", ls="--"); ax.set_yticks(y, F.metric + " | " + F.outcome + " | ctrl=" + F.control); ax.set_xscale("log")
ax.set_xlabel("effect per +0.1 Q (odds ratio / eps* ratio), 95% CI"); ax.set_title("Quality effect before vs after margin control")
fig.savefig(FIG / f"forest_{tag}.png", dpi=150, bbox_inches="tight")
print(f"\nSaved tables -> {TAB}  figures -> {FIG}  (suffix _{tag})")
