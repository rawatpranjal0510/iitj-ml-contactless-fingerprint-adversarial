import argparse, json, time, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
from config import OUT, FAR, BATCH
from src.dataset import Imgs
from src.model import Embedder, ArcFace, load_model
from src.pairs import genuine, impostor
from src.evaluate import embed_ds, full_array, pair_scores, eer, tar_at_far, boot_ci

p = argparse.ArgumentParser()
p.add_argument("--tag", default="v3")
p.add_argument("--select", default="eer", choices=["eer", "tar"])   # checkpoint selection on CROSS val pairs
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--lr_bb", type=float, default=1e-4)       # pretrained backbone: small LR
p.add_argument("--lr_head", type=float, default=1e-3)     # new layers (embedding fc + ArcFace): larger LR
p.add_argument("--wd", type=float, default=0.05)          # AdamW decoupled weight decay (old run: 1e-4 ~ none)
p.add_argument("--aug", default="strong", choices=["none", "light", "strong"])
p.add_argument("--dropout", type=float, default=0.3)
p.add_argument("--bn_emb", type=int, default=1)
p.add_argument("--smooth", type=float, default=0.1)       # label smoothing
p.add_argument("--m", type=float, default=0.40)
p.add_argument("--s", type=float, default=30.0)
p.add_argument("--ramp", type=int, default=8)
p.add_argument("--patience", type=int, default=12)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--no_pretrained", action="store_true")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--n_imp_train", type=int, default=20000)
a = p.parse_args()
(OUT / "runs").mkdir(exist_ok=True)

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
df = pd.read_csv(OUT / "images.csv")
split = json.load(open(OUT / "split.json"))
labels = {f: i for i, f in enumerate(split["train"])}
tr = df.index[df.split == "train"].tolist(); va = df.index[df.split == "val"].tolist()
vpairs = pd.read_csv(OUT / "pairs_val.csv")
cm = ((vpairs.kind == "cross") | (vpairs.label == 0)).values   # cross genuine + (cross-session) impostors
rng = np.random.default_rng(a.seed)
d_tr = df[df.split == "train"]
tpairs = pd.concat([genuine(d_tr), impostor(d_tr, a.n_imp_train, rng)], ignore_index=True)   # for TRAIN EER (no aug)

train_ds = Imgs(df, tr, labels, aug=a.aug)
tr_eval, va_eval = Imgs(df, tr), Imgs(df, va)             # cached once, no augmentation
loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, drop_last=True,
                    num_workers=a.workers, persistent_workers=a.workers > 0)

cfg = dict(dropout=a.dropout, bn=bool(a.bn_emb))
model = Embedder(pretrained=not a.no_pretrained, **cfg).to(dev)
head = ArcFace(128, len(labels), s=a.s, m=a.m).to(dev)
bb = [q for n, q in model.named_parameters() if not n.startswith("net.fc")]
fc = [q for n, q in model.named_parameters() if n.startswith("net.fc")]
opt = torch.optim.AdamW([dict(params=bb, lr=a.lr_bb, weight_decay=a.wd),
                         dict(params=fc, lr=a.lr_head, weight_decay=a.wd),
                         dict(params=head.parameters(), lr=a.lr_head, weight_decay=0.0)])
def lam(ep):                                              # linear warmup, then cosine decay
    if ep < a.warmup: return (ep + 1) / a.warmup
    return 0.5 * (1 + np.cos(np.pi * (ep - a.warmup) / max(1, a.epochs - a.warmup)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
ce = nn.CrossEntropyLoss(label_smoothing=a.smooth)

def evaluate():
    Ev = full_array(embed_ds(model, va_eval, dev), va, len(df))
    Et = full_array(embed_ds(model, tr_eval, dev), tr, len(df))
    S, y = pair_scores(Ev, vpairs), vpairs.label.values
    tar, tau = tar_at_far(S, y, FAR)
    r = dict(val_eer=eer(S, y), val_tar=tar,
             cross_eer=eer(S[cm], y[cm]), cross_tar=tar_at_far(S[cm], y[cm], FAR)[0],
             train_eer=eer(pair_scores(Et, tpairs), tpairs.label.values),
             gen_mean=float(S[y == 1].mean()), imp_mean=float(S[y == 0].mean()))
    for k in ["within_s1", "within_s2", "cross"]:
        r["tar_" + k] = float((S[(vpairs.kind == k).values] > tau).mean())
    return r

log, best, best_ep, bad = [], -np.inf, -1, 0
ck = OUT / "ckpt" / f"{a.tag}.pt"
for ep in range(a.epochs):
    model.train(); t0 = time.time(); tot = 0
    m = a.m * min(1.0, (ep + 1) / a.ramp)
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        loss = ce(head(model(x), y, m), y)
        opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()
    sched.step()
    r = dict(epoch=ep, m=round(m, 3), loss=tot / len(loader), **evaluate()); log.append(r)
    score = -r["cross_eer"] if a.select == "eer" else r["cross_tar"]      # higher is better
    print(f"ep{ep:02d} loss={r['loss']:.3f} trEER={r['train_eer']*100:.2f}% valEER={r['val_eer']*100:.2f}% "
          f"crossEER={r['cross_eer']*100:.2f}% crossTAR@{FAR:g}={r['cross_tar']*100:.1f}% "
          f"(s1 {r['tar_within_s1']*100:.0f} / s2 {r['tar_within_s2']*100:.0f}) {time.time()-t0:.0f}s"
          f"{'  *' if score > best else ''}", flush=True)
    if score > best:
        best, best_ep, bad = score, ep, 0
        torch.save({"state": model.state_dict(), "cfg": cfg}, ck)
    else:
        bad += 1
        if bad >= a.patience: print("early stop"); break
pd.DataFrame(log).to_csv(OUT / "runs" / f"{a.tag}.csv", index=False)

m_best = load_model(ck, dev)                              # final report on CROSS val pairs, cluster-bootstrap CIs
Ev = full_array(embed_ds(m_best, va_eval, dev), va, len(df))
P = vpairs.copy(); P["S"] = pair_scores(Ev, P)
Pc = P[cm].reset_index(drop=True)
ci_tar = boot_ci(Pc, lambda s, y: tar_at_far(s, y, FAR)[0]); ci_eer = boot_ci(Pc, eer)
res = dict(tag=a.tag, args=vars(a), select=a.select, best_epoch=best_ep,
           cross_val_eer=eer(Pc.S.values, Pc.label.values), cross_val_tar=tar_at_far(Pc.S.values, Pc.label.values, FAR)[0],
           all_val_eer=eer(P.S.values, P.label.values), ci95_tar=ci_tar, ci95_eer=ci_eer)
json.dump(res, open(OUT / "runs" / f"{a.tag}.json", "w"), indent=1)
print(f"\n[{a.tag}] select={a.select} best epoch {best_ep} | cross val EER = {res['cross_val_eer']*100:.2f}%  "
      f"95% CI {ci_eer[0]*100:.2f}-{ci_eer[1]*100:.2f} | cross val TAR@FAR{FAR:g} = {res['cross_val_tar']*100:.1f}%  "
      f"95% CI {ci_tar[0]*100:.1f}-{ci_tar[1]*100:.1f}")
