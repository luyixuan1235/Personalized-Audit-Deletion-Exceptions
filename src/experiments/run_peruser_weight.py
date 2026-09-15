"""检验2: per-user 类别加权 (而非全局). 机制预测: 应当保护大方用户的罕见拒绝."""
import numpy as np, torch, sys, os
sys.path.insert(0,'.')
import metrics as M
from data import build_records, per_user_folds
from collections import defaultdict

FIELDS=("user_id","domain","tool","data_type")
def fm(tr,te,mode,dim=16,epochs=150,lr=0.05,wd=1e-6,seed=42):
    """mode: none=普通BCE | global=全局类别权重 | peruser=按每个用户自己的不平衡度加权"""
    torch.manual_seed(seed); np.random.seed(seed)
    idx={f:{} for f in FIELDS}
    for r in tr:
        for f in FIELDS: idx[f].setdefault(r[f], len(idx[f])+1)
    sizes=[len(idx[f])+1 for f in FIELDS]
    enc=lambda R: torch.tensor([[idx[f].get(r[f],0) for f in FIELDS] for r in R],dtype=torch.long)
    offs=torch.tensor(np.cumsum([0]+sizes[:-1]),dtype=torch.long)
    tot=int(sum(sizes))
    lin=torch.nn.Embedding(tot,1); torch.nn.init.zeros_(lin.weight)
    fac=torch.nn.Embedding(tot,dim); torch.nn.init.normal_(fac.weight,std=0.01)
    b0=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam(list(lin.parameters())+list(fac.parameters())+[b0],lr=lr,weight_decay=wd)
    Xtr=enc(tr)+offs; ytr=torch.tensor([r["label"] for r in tr],dtype=torch.float32)

    # --- 样本权重
    if mode=="none":
        w=torch.ones(len(tr))
    elif mode=="global":
        npos=ytr.sum().item(); nneg=len(ytr)-npos
        w=torch.where(ytr>0.5, torch.tensor(1.0), torch.tensor(npos/max(nneg,1)))
    else:  # peruser: 每个用户内部, 少数类被上调到与多数类等权
        cnt=defaultdict(lambda:[0,0])
        for r in tr:
            c=cnt[r["user_id"]]; c[int(r["label"])]+=1
        w=torch.tensor([ (max(cnt[r["user_id"]])/max(cnt[r["user_id"]][int(r["label"])],1))
                         for r in tr], dtype=torch.float32)
        w=w/w.mean()
    def logits(X):
        l=lin(X).sum(1).squeeze(-1)+b0
        v=fac(X); s=v.sum(1)
        return l+0.5*((s*s).sum(-1)-(v*v).sum(-1).sum(-1))
    for _ in range(epochs):
        opt.zero_grad()
        z=logits(Xtr)
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(z,ytr,reduction='none')*w).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return (torch.sigmoid(logits(enc(te)+offs)).numpy(),
                torch.sigmoid(logits(Xtr)).numpy())

records,_=build_records(); folds=per_user_folds(records,5,42)
print(f"{'加权方式':14s}{'Acc':>7}{'聚合FGR':>9}{'保守':>8}{'大方':>8}{'比值':>7}{'保守grant被误拒':>16}")
for mode,lab in (("none","无加权(BCE)"),("global","全局类别权重"),("peruser","per-user 权重")):
    G,V,Gg,Vg,ACC=[],[],[],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        p,ptr=fm(tr,te,mode)
        y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
        thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
        u=np.array([r["user_id"] for r in te])
        gr={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
        Gu=np.array([gr[uu] for uu in u])
        dn=(y==0); gt=(y==1)
        G+=list(Gu[dn]); V+=list(yh[dn].astype(float))
        Gg+=list(Gu[gt]); Vg+=list((1-yh[gt]).astype(float))
        ACC.append(100*(yh==y).mean())
    G,V,Gg,Vg=map(np.array,(G,V,Gg,Vg))
    lo=100*V[G<0.2].mean(); hi=100*V[G>=0.6].mean()
    rg=100*Vg[Gg<0.2].mean() if (Gg<0.2).any() else float('nan')
    print(f"{lab:14s}{np.mean(ACC):7.1f}{100*V.mean():9.1f}{lo:7.1f}%{hi:7.1f}%{hi/max(lo,.1):6.1f}x{rg:14.1f}%")
