import argparse, json, numpy as np, pandas as pd, torch
from config import OUT, FAR, FAR_FLOOR
from src.model import load_model
from src.evaluate import embed_rows, pair_scores, eer, tau_at_far

ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", default="v3"); a = ap.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(OUT / "images.csv")
model = load_model(OUT / "ckpt" / f"{a.ckpt}.pt", dev)

for split in ["val", "test"]:
    rows = df.index[df.split == split].tolist()
    P = pd.read_csv(OUT / f"pairs_{split}.csv")
    P["S"] = pair_scores(embed_rows(model, df, rows, dev), P)
    if split == "val":                                   # both thresholds are frozen from VAL impostors only
        imp = P.S[P.label == 0].values
        tau, tau_floor = tau_at_far(imp, FAR), tau_at_far(imp, FAR_FLOOR)
        json.dump(dict(tau=tau, far=FAR, tau_floor=tau_floor, far_floor=FAR_FLOOR, ckpt=a.ckpt),
                  open(OUT / "tau.json", "w"), indent=1)
    P["M"] = P.S - tau
    P["accepted"] = P.S > tau
    P.to_csv(OUT / f"scores_{split}.csv", index=False)
    g, i = P[P.label == 1], P[P.label == 0]
    print(f"\n[{split}] EER={eer(P.S.values, P.label.values)*100:.2f}%  tau={tau:.4f}  tau_floor={tau_floor:.4f}")
    for name, t in [("tau", tau), ("tau_floor", tau_floor)]:
        print(f"  {name:9s}: FAR = {(i.S > t).mean()*100:.3f}%   TAR = {(g.S > t).mean()*100:.1f}%  "
              f"({int((g.S > t).sum())} of {len(g)} genuine)   TAR cross = {(g.S[g.kind == 'cross'] > t).mean()*100:.1f}%")
    print("  TAR@tau by pair kind:", (g.groupby('kind').accepted.mean()*100).round(1).to_dict())
