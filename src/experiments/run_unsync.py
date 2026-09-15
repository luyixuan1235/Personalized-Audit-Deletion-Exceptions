"""可证伪预言 (Zhu & Caverlee ECIR'24, unsynchronized learning):
例外是否需要更多 epoch 才收敛? 追踪 routine vs exception 的覆盖率随 epoch 的变化."""
import numpy as np, torch, sys
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M
from data import build_records, per_user_folds
FIELDS=("user_id","domain","tool","data_type")
records,_=build_records(); folds=per_user_folds(records,5,42)
EPOCHS=[10,25,50,100,150,250,400,600,1000]

curves={'routine':defaultdict(list),'exception':defaultdict(list),'acc':defaultdict(list)}
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    torch.manual_seed(42); np.random.seed(42)
    idx={fl:{} for fl in FIELDS}
    for r in tr:
        for fl in FIELDS: idx[fl].setdefault(r[fl], len(idx[fl])+1)
    sizes=[len(idx[fl])+1 for fl in FIELDS]
    enc=lambda R: torch.tensor([[idx[fl].get(r[fl],0) for fl in FIELDS] for r in R],dtype=torch.long)
    offs=torch.tensor(np.cumsum([0]+sizes[:-1]),dtype=torch.long); tot=int(sum(sizes))
    lin=torch.nn.Embedding(tot,1); torch.nn.init.zeros_(lin.weight)
    fac=torch.nn.Embedding(tot,16); torch.nn.init.normal_(fac.weight,std=0.01)
    b0=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam(list(lin.parameters())+list(fac.parameters())+[b0],lr=0.05,weight_decay=1e-6)
    Xtr=enc(tr)+offs; Xte=enc(te)+offs
    ytr=torch.tensor([r["label"] for r in tr],dtype=torch.float32)
    y=np.array([r["label"] for r in te]); ytr_np=np.array([r["label"] for r in tr])
    u=np.array([r["user_id"] for r in te])
    ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
    mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u]); exc=(y!=mu)
    def z(X):
        l=lin(X).sum(1).squeeze(-1)+b0; v=fac(X); s=v.sum(1)
        return l+0.5*((s*s).sum(-1)-(v*v).sum(-1).sum(-1))
    bce=torch.nn.BCEWithLogitsLoss()
    ep=0
    for target in EPOCHS:
        while ep<target:
            opt.zero_grad(); bce(z(Xtr),ytr).backward(); opt.step(); ep+=1
        with torch.no_grad():
            p=torch.sigmoid(z(Xte)).numpy(); ptr=torch.sigmoid(z(Xtr)).numpy()
        thr=float(M.equal_error_threshold(ytr_np,ptr)); yh=(p>=thr).astype(int)
        ov=(yh!=y)
        curves['routine'][target].append(100*ov[~exc].mean())
        curves['exception'][target].append(100*ov[exc].mean())
        curves['acc'][target].append(100*(yh==y).mean())

print("追踪 override rate 随训练 epoch 的变化 (5折均值)\n")
print(f"{'epoch':>7}{'Acc':>8}{'routine 覆盖':>14}{'exception 覆盖':>16}")
best_r=best_e=(9e9,None)
for t in EPOCHS:
    r_=np.mean(curves['routine'][t]); e_=np.mean(curves['exception'][t]); a=np.mean(curves['acc'][t])
    if r_<best_r[0]: best_r=(r_,t)
    if e_<best_e[0]: best_e=(e_,t)
    print(f"{t:>7}{a:8.1f}{r_:13.1f}%{e_:15.1f}%")
print(f"\nroutine  最优 epoch = {best_r[1]}  ({best_r[0]:.1f}%)")
print(f"exception 最优 epoch = {best_e[1]}  ({best_e[0]:.1f}%)")
print("=> " + ("✓ unsynchronized learning: 例外需要更多 epoch 才达峰"
               if best_e[1]>best_r[1] else
               "✗ 例外并不需要更多 epoch —— 该预言不成立"))
