import numpy as np, pandas as pd, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, json
from config import OUT
from src.evaluate import eer
tau = json.load(open(OUT / "tau.json"))["tau"]
for split in ["val", "test"]:
    P = pd.read_csv(OUT / f"scores_{split}.csv")
    imp = P[P.label == 0].S.values
    print(f"[{split}] EER by genuine pair kind (same impostors):")
    for k in ["within_s1", "within_s2", "cross"]:
        g = P[P.kind == k].S.values
        print(f"   {k:10s} EER={eer(np.r_[g, imp], np.r_[np.ones(len(g)), np.zeros(len(imp))])*100:5.2f}%   "
              f"mean S={g.mean():.3f}   TAR={np.mean(g > tau)*100:5.1f}%")
    print(f"   impostor   mean S={imp.mean():.3f}  99.9th pct={np.quantile(imp, .999):.3f}")
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(-0.4, 1, 80)
    ax.hist(imp, bins, density=True, alpha=.5, label="impostor", color="tab:red")
    for k, c in [("within_s1", "tab:green"), ("cross", "tab:blue")]:
        ax.hist(P[P.kind == k].S, bins, density=True, alpha=.5, label=f"genuine {k}", color=c)
    ax.axvline(tau, color="k", ls="--", label="tau (FAR 0.1%)"); ax.set_xlabel("cosine score S"); ax.legend()
    fig.savefig(OUT / "figs" / f"score_hist_{split}.png", dpi=150, bbox_inches="tight")