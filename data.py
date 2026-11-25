import re, json
import numpy as np, pandas as pd
from config import DATA_ROOT, OUT, SEED, N_VAL, N_TEST

SESSIONS = {"first_session": 1, "second_session": 2}

def build_table():
    rows = []
    for sname, snum in SESSIONS.items():
        for fdir in sorted((DATA_ROOT / sname).iterdir()):
            if not (fdir.is_dir() and re.fullmatch(r"p\d+", fdir.name)):
                continue                                    # skips copy folders like "p26 (1)"
            for img in sorted(fdir.iterdir()):
                m = re.fullmatch(r"p([1-6])\.bmp", img.name)
                if m:                                       # skips copies like "p6 (1).bmp"
                    rows.append(dict(path=str(img.relative_to(DATA_ROOT)),
                                     finger=int(fdir.name[1:]), session=snum, sample=int(m.group(1))))
    return pd.DataFrame(rows)

def make_split(df):
    rng = np.random.default_rng(SEED)
    n_sess = df.groupby("finger").session.nunique()
    two = rng.permutation(n_sess[n_sess == 2].index.to_numpy())
    one = n_sess[n_sess == 1].index.to_numpy()
    val, test = two[:N_VAL], two[N_VAL:N_VAL + N_TEST]
    train = np.concatenate([two[N_VAL + N_TEST:], one])
    assert len(set(train) | set(val) | set(test)) == len(train) + len(val) + len(test) == df.finger.nunique()
    return {k: sorted(map(int, v)) for k, v in dict(train=train, val=val, test=test).items()}

if __name__ == "__main__":
    df = build_table()
    print("images:", len(df), "| fingers:", df.finger.nunique(), "| per session:", df.session.value_counts().to_dict())
    split = make_split(df)
    df["split"] = df.finger.map({f: k for k, v in split.items() for f in v})
    print("fingers per split:", {k: len(v) for k, v in split.items()})
    print("images per split:", df.split.value_counts().to_dict())
    df.to_csv(OUT / "images.csv", index=False)
    json.dump(split, open(OUT / "split.json", "w"))