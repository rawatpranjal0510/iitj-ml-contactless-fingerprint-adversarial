import numpy as np, pandas as pd
from itertools import combinations
from config import OUT, SEED

def genuine(df):
    rows = []
    for f, g in df.groupby("finger"):
        s1 = g[g.session == 1].index.tolist()
        s2 = g[g.session == 2].index.tolist()
        for a, b in combinations(s1, 2): rows.append((a, b, f, "within_s1"))
        for a, b in combinations(s2, 2): rows.append((a, b, f, "within_s2"))
        for a in s1:
            for b in s2: rows.append((a, b, f, "cross"))     # enrolled = session 1, probe = session 2
    out = pd.DataFrame(rows, columns=["enr", "probe", "finger", "kind"])
    out["label"] = 1
    return out

def impostor(df, n, rng):                                     # random any-session impostors (train-set EER only)
    idx = df.index.to_numpy()
    fing = df.finger.to_numpy()
    seen, rows = set(), []
    while len(rows) < n:
        a, b = rng.choice(len(idx), 2, replace=False)
        if fing[a] == fing[b] or (a, b) in seen:
            continue
        seen.add((a, b))
        rows.append((idx[a], idx[b], fing[b], "impostor"))   # finger = probe's finger (cluster id)
    out = pd.DataFrame(rows, columns=["enr", "probe", "finger", "kind"])
    out["label"] = 0
    return out

def impostor_cross(df):
    """ALL cross-session impostors: enrolled = session 1, probe = session 2, different fingers.
    Matches the deployment scenario, so tau frozen on these controls the FAR we actually attack."""
    a, b = np.meshgrid(df.index[df.session == 1], df.index[df.session == 2], indexing="ij")
    a, b = a.ravel(), b.ravel()
    fa, fb = df.finger.loc[a].to_numpy(), df.finger.loc[b].to_numpy()
    keep = fa != fb
    out = pd.DataFrame(dict(enr=a[keep], probe=b[keep], finger=fb[keep], kind="impostor"))
    out["label"] = 0
    return out

if __name__ == "__main__":
    df = pd.read_csv(OUT / "images.csv")
    for split in ["val", "test"]:
        d = df[df.split == split]
        g, i = genuine(d), impostor_cross(d)
        pairs = pd.concat([g, i], ignore_index=True)
        pairs.to_csv(OUT / f"pairs_{split}.csv", index=False)      # enr/probe are row indices of images.csv
        print(split, "| genuine:", len(g), g.kind.value_counts().to_dict(), "| impostor (cross-session):", len(i))
