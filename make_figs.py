import json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from config import OUT, FAR, FAR_FLOOR
from src.evaluate import tau_at_far
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5, "legend.fontsize": 8, "savefig.dpi": 300, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})
FIG, TAB = OUT / "figs" / "paper", OUT / "tables"; FIG.mkdir(parents=True, exist_ok=True)
C = {1: "#d62728", 2: "#ff7f0e", 3: "#2ca02c"}; SPL = ["test", "val"]; PRI = "0.05"
T = json.load(open(OUT / "tau.json")); tau, tf = T["tau"], T["tau_floor"]
def save(fig, name): [fig.savefig(FIG / f"{name}.{e}") for e in ("png", "pdf")]; plt.close(fig); print("saved", name)
def acc(s): A = pd.read_csv(OUT / f"attacks_{s}.csv"); return A[A.acc & ~A.nfiq2_fail]

# Fig 1: score distributions
fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, s in zip(axs, SPL):
    P = pd.read_csv(OUT / f"scores_{s}.csv"); bins = np.linspace(-0.4, 1, 85)
    ax.hist(P.S[P.label == 0], bins, density=True, alpha=.5, color="tab:red", label="impostor (cross-session)")
    for k_, c, l in [("within_s1", "tab:green", "genuine same-session (S1)"), ("cross", "tab:blue", "genuine cross-session")]:
        ax.hist(P.S[P.kind == k_], bins, density=True, alpha=.5, color=c, label=l)
    ax.axvline(tau, c="k", ls="--", lw=1, label="τ_fixed (FAR 0.1%)"); ax.axvline(tf, c="r", ls=":", lw=1.2, label="τ_floor (FAR 0.2%)")
    ax.set(xlabel="cosine score S", title=f"{s}: matcher score distributions")
axs[0].set_ylabel("density"); axs[0].legend(loc="upper left"); save(fig, "fig1_score_hist")

# Fig 2: ASR vs budget by tier + B3
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, s in zip(axs, SPL):
    d = acc(s); A = pd.read_csv(OUT / f"attacks_{s}.csv"); A = A[A.acc]
    E = sorted(float(c[5:]) for c in d.columns if c.startswith("succ_"))
    ax.axvspan(0.6, 10, color="0.93", zorder=0); ax.text(2.6, 45, "proposal grid\n{1,2,4,8}/255", ha="center", fontsize=8, color="0.35")
    for t in (1, 2, 3): g = d[d.tier_nfiq2 == t]; ax.plot(E, [g[f"succ_{e:g}"].mean() * 100 for e in E], "o-", c=C[t], ms=4, label=f"PGD, Q{t} (n={len(g)})")
    ax.plot(E, [A[f"succn_{e:g}"].mean() * 100 for e in E], "k--", label="random sign noise (B3)")
    ax.set(xscale="log", xlabel="L∞ budget ε (×1/255)", title=f"{s}: attack success on accepted cross pairs")
axs[0].set_ylabel("ASR (%)"); axs[0].legend(loc="lower center", bbox_to_anchor=(0.55, 0.06)); save(fig, "fig2_asr_vs_eps")

# Fig 3: ASR within margin bands
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
for ax, s in zip(axs, SPL):
    B = pd.read_csv(TAB / f"bands_{s}.csv"); B = B[np.isclose(B.eps, 0.05)]; bands = list(dict.fromkeys(B.band)); x = np.arange(len(bands)); w = .27
    ax.axvspan(-0.5, 1.5, color="0.93", zorder=0); ax.text(0.5, 111, "saturated: all tiers ≈ 100%", ha="center", fontsize=8, color="0.4")
    for i, t in enumerate((1, 2, 3)):
        b = B[B.tier == f"Q{t}"].set_index("band").reindex(bands); xx = x + (i - 1) * w
        ax.bar(xx, b.ASR, w, color=C[t], label=f"Q{t}", yerr=[(b.ASR - b.lo).clip(0), (b.hi - b.ASR).clip(0)], capsize=2, error_kw=dict(lw=.8))
        for xi, asr, hi, n in zip(xx, b.ASR.fillna(0), b.hi.fillna(0), b.n): ax.text(xi, max(asr, hi) + 2, f"{n}", ha="center", fontsize=6.5, color="0.25")
    ax.set_xticks(x, [bb.replace(">=", "≥ ") for bb in bands]); ax.set_ylim(0, 118)
    ax.set(xlabel="clean margin band  M = S − τ", title=s)
axs[0].set_ylabel("ASR (%), 95% finger-clustered CI"); axs[1].legend(loc="upper right")
fig.suptitle(f"ASR at ε = {PRI}/255 within clean-margin bands (numbers above bars = pairs)", y=1.01); save(fig, "fig3_asr_bands")

# Fig 4: eps* vs margin with per-tier fits
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
for ax, s in zip(axs, SPL):
    d = acc(s)
    for t in (3, 2, 1): g = d[d.tier_nfiq2 == t]; ax.scatter(g.M, g.eps_star, s=7, alpha=.4, c=C[t], lw=0, label=f"Q{t} (n={len(g)})")
    for t in (1, 2, 3):
        g = d[d.tier_nfiq2 == t]; b1, b0 = np.polyfit(np.log(g.M), np.log(g.eps_star), 1); xs = np.geomspace(g.M.min(), g.M.max(), 50)
        ax.plot(xs, np.exp(b0) * xs ** b1, c=C[t], lw=2.2)
    ax.set(xscale="log", yscale="log", xlabel="clean margin  M = S − τ",
           title=f"{s} (Spearman ρ = {d.eps_star.corr(d.M, method='spearman'):.2f})")
axs[0].set_ylabel("ε* (×1/255): minimal PGD budget to cross τ"); axs[0].legend(loc="upper left", markerscale=2.5)
fig.suptitle("Robustness radius ε* vs clean margin (lines = per-tier log-log fit)", y=1.01); save(fig, "fig4_epsstar_vs_M")

# Fig 5 / S1: forest plots (test vs val)
LAB = {"none": "no margin control", "M": "+ linear M", "bandM": "+ binned M", "logM": "+ log M", "FE": "+ log M + finger FE"}
ALL = list(LAB)
def forest(metric, name):
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.9)); fig.subplots_adjust(wspace=0.06)
    for ax, outc, ttl, xl in [(axs[0], "log_eps*", "robustness radius ε*", "ε* ratio per +0.1 Q\n(> 1: better quality = more robust)"),
                              (axs[1], f"success@{PRI}", f"attack success at ε = {PRI}/255", "odds ratio per +0.1 Q\n(< 1: better quality = fewer successes)")]:
        rows = ALL if outc == "log_eps*" else ALL[:4]
        for j, (s_, mk, col) in enumerate([("test", "o", "k"), ("val", "s", "tab:blue")]):
            R = pd.read_csv(TAB / f"regression_{s_}.csv"); R = R[(R.metric == metric) & (R.outcome == outc)].set_index("control")[["ratio", "lo", "hi"]]
            if outc == "log_eps*":
                F = pd.read_csv(TAB / f"finger_fe_{s_}.csv"); f_ = F[(F.metric == metric) & (F.model == "logM+finger_FE")].iloc[0]; R.loc["FE"] = [f_.ratio, f_.lo, f_.hi]
            r = R.loc[rows]; y = np.array([len(ALL) - 1 - ALL.index(c) for c in rows]) + (0.14 if j == 0 else -0.14)
            ax.errorbar(r.ratio, y, xerr=[(r.ratio - r.lo).clip(0), (r.hi - r.ratio).clip(0)], fmt=mk, c=col, capsize=3, ms=5, label=s_)
        ax.axvline(1, c="gray", ls="--", lw=1); ax.set_yticks(np.arange(len(ALL))[::-1], [LAB[c] for c in ALL]); ax.set_ylim(-0.6, len(ALL) - 0.4)
        ax.set(title=ttl, xlabel=xl)
    axs[1].tick_params(labelleft=False); axs[0].legend(loc="lower right")
    fig.suptitle(f"{metric.upper()}: quality effect before vs after margin control (95% CI: finger-clustered bootstrap; finger-FE row: cluster-robust SE)", y=1.02, fontsize=10.5)
    save(fig, name)
forest("nfiq2", "fig5_forest_nfiq2"); forest("mclfiq", "figS1_forest_mclfiq")

# Policy figures: reproduce the val selection rule of policy.py from the saved grid
Gr = pd.read_csv(TAB / "policy_grid_val_nfiq2.csv"); Gl = pd.read_csv(TAB / "policy_global_sweep_val_nfiq2.csv")
budget, obj = (FAR + FAR_FLOOR) / 2 * 100, f"TARatt_{PRI}_Q1"
best = Gr[Gr.FAR <= budget].sort_values([obj, "alpha"], ascending=[False, True]).iloc[0]; al, qh = best.alpha, best.qh
Pv = pd.read_csv(OUT / "scores_val.csv"); tau_g = float(np.clip(tau_at_far(Pv.S[Pv.label == 0].values, best.FAR / 100), tf, tau))
cut = json.load(open(OUT / "quality_cutoffs.json"))["nfiq2"]

# Fig 6: tau(Q)
qs = np.linspace(0, 1, 401); fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(qs, np.clip(tau - al * np.maximum(0, qh - qs), tf, tau), "b-", lw=2, label=f"B2 τ(Q): α = {al:.3f}, Q_high = {qh:.2f}")
for y, l, c, ls in [(tau, "τ_fixed (FAR 0.1%)", "k", "--"), (tau_g, "B2-global (equal val FAR)", "g", "-."), (tf, "τ_floor (FAR 0.2%)", "r", ":")]: ax.axhline(y, c=c, ls=ls, lw=1.2, label=l)
for x0, x1, l in [(0, cut["c1"], "Q1"), (cut["c1"], cut["c2"], "Q2"), (cut["c2"], 1, "Q3")]: ax.text((x0 + x1) / 2, tau + 0.003, l, ha="center", fontsize=9, color="0.3")
for c_ in (cut["c1"], cut["c2"]): ax.axvline(c_, c="gray", lw=.8, ls=":")
ax.set(xlabel="probe quality Q (NFIQ2, normalised)", ylabel="acceptance threshold", ylim=(tf - 0.006, tau + 0.008), title="Quality-adaptive threshold (tuned on validation)")
ax.legend(loc="lower right"); save(fig, "fig6_tau_of_Q")

# Fig 7: validation frontier, clean and under attack
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, ttl in [(axs[0], "TAR_Q1", "clean"), (axs[1], obj, f"under PGD attack (ε = {PRI}/255)")]:
    for n in ["c1", "c2", "q90"]: g = Gr[Gr.qh_name == n]; ax.plot(g.FAR, g[col], label=f"adaptive τ(Q), Q_high = {g.qh.iloc[0]:.2f} ({n})")
    ax.plot(Gl.FAR, Gl[col], "k--", label="single global threshold"); ax.plot(best.FAR, best[col], "r*", ms=13, zorder=5, label="chosen B2")
    ax.axvline(budget, c="gray", ls=":", label="tuning FAR budget (0.15%)"); ax.set(xlabel="validation FAR (%)", ylabel="Q1 TAR (%)", title=f"Validation frontier for Q1 probes: {ttl}")
axs[0].legend(loc="lower right"); save(fig, "fig7_policy_frontier_val")

# Fig 8: test TAR under attack by policy
NM = dict(B1_fixed="B1 fixed τ", B2_adaptive="B2 adaptive τ(Q)", B2_global="B2-global (equal val FAR)", floor_global="τ_floor for everyone")
R = pd.read_csv(TAB / "policy_results_nfiq2.csv"); R = R[R.split == "test"].set_index("policy")
E = sorted(float(c[7:]) for c in R.columns if c.startswith("TARatt_") and not c.endswith("_Q1"))
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
for ax, suf, ttl in [(axs[0], "", "all genuine cross pairs"), (axs[1], "_Q1", "Q1 (low-quality) probes")]:
    for p_, st in zip(NM, ["o-", "s-", "^--", "d:"]): ax.plot(E, [R.loc[p_, f"TARatt_{e:g}{suf}"] for e in E], st, ms=4, label=f"{NM[p_]} (test FAR {R.loc[p_, 'FAR']:.2f}%)")
    ax.set(xscale="log", xlabel="L∞ budget ε (×1/255)", title=f"test: TAR under PGD attack, {ttl}")
axs[0].set_ylabel("TAR (%)"); axs[0].legend(); save(fig, "fig8_policy_tar_vs_eps_test")

# Fig 9: B4 hard rejection
B4 = pd.read_csv(TAB / "b4_test_nfiq2.csv"); fig, ax = plt.subplots(figsize=(5.2, 4.4))
for r_, c, l in [("quality", "tab:blue", "reject low quality (request recapture)"), ("margin", "tab:purple", "reject small margin (raise τ)")]:
    b = B4[B4.rule == r_]; ax.plot(b.genuine_rejected, b.attacks_prevented, c=c, lw=2, label=l)
ax.plot([0, 100], [0, 100], "k:", label="random rejection")
ax.set(xlabel="genuine users rejected (% of accepted genuine pairs)", ylabel="successful attacks prevented (%)", xlim=(0, 100), ylim=(0, 100),
       title=f"B4: hard rejection trade-off (test, ε = {PRI}/255)"); ax.legend(loc="lower right"); save(fig, "fig9_b4_rejection")
