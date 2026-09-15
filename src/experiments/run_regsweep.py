"""机制导出的解药: 效应是'正则化 + 欠训练'的产物吗?
在真实数据上扫 weight_decay x epochs, 看大方用户的罕见拒绝是否被拯救."""
import numpy as np, torch, sys
sys.path.insert(0,'.')
import metrics as M
_OUT={}
from data import build_records, per_user_folds
FIELDS=("user_id","domain","tool","data_type")

def fm(tr,te,dim=16,epochs=150,lr=0.05,wd=1e-6,seed=42):
    torch.manual_seed(seed); np.random.seed(seed)
    idx={f:{} for f in FIELDS}
    for r in tr:
        for f in FIELDS: idx[f].setdefault(r[f], len(idx[f])+1)
    sizes=[len(idx[f])+1 for f in FIELDS]
    enc=lambda R: torch.tensor([[idx[f].get(r[f],0) for f in FIELDS] for r in R],dtype=torch.long)
    offs=torch.tensor(np.cumsum([0]+sizes[:-1]),dtype=torch.long); tot=int(sum(sizes))
    lin=torch.nn.Embedding(tot,1); torch.nn.init.zeros_(lin.weight)
    fac=torch.nn.Embedding(tot,dim); torch.nn.init.normal_(fac.weight,std=0.01)
    b0=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam(list(lin.parameters())+list(fac.parameters())+[b0],lr=lr,weight_decay=wd)
    Xtr=enc(tr)+offs; ytr=torch.tensor([r["label"] for r in tr],dtype=torch.float32)
    def logits(X):
        l=lin(X).sum(1).squeeze(-1)+b0; v=fac(X); s=v.sum(1)
        return l+0.5*((s*s).sum(-1)-(v*v).sum(-1).sum(-1))
    bce=torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad(); bce(logits(Xtr),ytr).backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(logits(enc(te)+offs)).numpy(), torch.sigmoid(logits(Xtr)).numpy()

records,_=build_records(); folds=per_user_folds(records,5,42)
print("真实数据 (Wu). 机制预测: 正则越弱 / 训练越久 -> 大方用户越被保护\n")
print(f"{'wd':>8}{'epochs':>8}{'Acc':>7}{'聚合FGR':>9}{'保守':>8}{'大方':>8}{'比值':>7}")
for wd in (1e-2, 1e-4, 1e-6, 0.0):
    for ep in (150, 600, 2000):
        G,V,ACC=[],[],[]
        for f in range(5):
            tr=[records[i] for i in range(len(records)) if folds[i]!=f]
            te=[records[i] for i in range(len(records)) if folds[i]==f]
            p,ptr=fm(tr,te,epochs=ep,wd=wd)
            y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
            thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
            u=np.array([r["user_id"] for r in te])
            gr={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
            Gu=np.array([gr[uu] for uu in u]); dn=(y==0)
            G+=list(Gu[dn]); V+=list(yh[dn].astype(float)); ACC.append(100*(yh==y).mean())
        G,V=np.array(G),np.array(V)
        lo=100*V[G<0.2].mean(); hi=100*V[G>=0.6].mean()
        star=" <<<" if (wd<=1e-6 and ep==2000) else ""
        print(f"{wd:8.0e}{ep:8d}{np.mean(ACC):7.1f}{100*V.mean():9.1f}{lo:7.1f}%{hi:7.1f}%{hi/max(lo,.1):6.1f}x{star}", flush=True)
        _OUT[f"wd={wd:g},ep={ep}"]=dict(acc=float(np.mean(ACC)),refusal_override=float(100*V.mean()),restrictive=float(lo),permissive=float(hi))

import json as _json, os as _os
_json.dump(_OUT, open(_os.path.join(_os.environ.get("PERM_RESULTS_DIR","../results"),"table49_regsweep.json"),"w"), indent=1)
print("saved table49_regsweep.json")
