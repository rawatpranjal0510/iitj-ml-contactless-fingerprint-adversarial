import argparse, json, zipfile, numpy as np, pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter
from config import DATA_ROOT, OUT

QDIR = OUT / "qimg"
METRICS = ["nfiq2", "mclfiq"]
SRC = {"nfiq2": "q_nfiq2.csv",          # NFIQ2 v2.3 (primary)
       "mclfiq": "q_mclfiq_v22.csv"}   # MCLFIQ model under NFIQ2 v2.2, the version it was trained with
def qname(r): return f"s{r.session}_f{r.finger}_p{r.sample}.png"          # for itertuples rows (prep)
def qnames(df): return "s" + df.session.astype(str) + "_f" + df.finger.astype(str) + "_p" + df["sample"].astype(str) + ".png"

def contactify(path, s=8):
    """Contactless -> contact-like for NFIQ2/MCLFIQ: local mean/std normalisation (sigma=s), invert (dark ridges).
    Native size: measured ridge period ~8.6 px is already ~500 PPI, so no rescaling."""
    a = np.asarray(Image.open(DATA_ROOT / path).convert("L"), float)
    d = a - gaussian_filter(a, s); z = d / np.sqrt(gaussian_filter(d * d, s) + 1e-6)
    return Image.fromarray(255 - np.clip(127.5 + 50 * z, 0, 255).astype(np.uint8))

def prep(df):
    QDIR.mkdir(exist_ok=True)
    with zipfile.ZipFile(OUT / "qimg.zip", "w") as zf:
        for r in df.itertuples():
            f = QDIR / qname(r); contactify(r.path).save(f, dpi=(500, 500)); zf.write(f, f.name)
    print(len(df), "images ->", OUT / "qimg.zip")

def merge(df):
    cuts = {}
    for m in METRICS:
        R = pd.read_csv(OUT / SRC[m])
        R["file"] = R.Filename.str.replace("\\", "/", regex=False).str.split("/").str[-1]
        raw = qnames(df).map(dict(zip(R.file, pd.to_numeric(R.QualityScore, errors="coerce"))))
        assert qnames(df).isin(R.file).all(), f"{m}: some images missing from {SRC[m]}"
        df[f"{m}_raw"] = raw
        df[f"{m}_fail"] = raw.isna() | (raw <= 0)                       # NA / 0 = extractor failure
        df[f"q_{m}"] = np.where(df[f"{m}_fail"], 0.0, raw / 100)        # Q in [0,1]; failures -> lowest quality
        ref = df.loc[(df.split == "val") & (df.session == 2), f"q_{m}"] # tier cutoffs from VAL probes only
        c1, c2 = np.quantile(ref, [1/3, 2/3])
        df[f"tier_{m}"] = 1 + (df[f"q_{m}"] > c1).astype(int) + (df[f"q_{m}"] > c2).astype(int)
        cuts[m] = dict(c1=float(c1), c2=float(c2), n_fail=int(df[f"{m}_fail"].sum()))
        print(f"\n[{m}] cutoffs (val S2 tertiles): Q1 <= {c1:.2f} < Q2 <= {c2:.2f} < Q3 | failures: {cuts[m]['n_fail']}")
        print(df.groupby(["split", "session"])[f"q_{m}"].describe()[["count", "mean", "std", "min", "50%", "max"]].round(3))
        print("tier counts (S2 probes):", df[df.session == 2].groupby("split")[f"tier_{m}"].value_counts().unstack().to_dict("index"))
    print("\nSpearman NFIQ2 vs MCLFIQ:", round(df.q_nfiq2.corr(df.q_mclfiq, method="spearman"), 3))
    df.to_csv(OUT / "quality.csv", index=False)
    json.dump(cuts, open(OUT / "quality_cutoffs.json", "w"), indent=1)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["prep", "merge"]); a = ap.parse_args()
    df = pd.read_csv(OUT / "images.csv")
    prep(df) if a.mode == "prep" else merge(df)
