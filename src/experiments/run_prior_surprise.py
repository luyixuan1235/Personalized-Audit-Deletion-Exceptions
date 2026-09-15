"""Exception-aware v2: 按'先验的误差'加权 —— 直接给同时违背两个先验的决策加权.
注意这不是 focal loss: focal 按模型自己的置信度加权, 而模型的置信度会收敛到先验, 自我抵消.
按固定先验的误差加权, 不会自我抵消."""
import numpy as np, torch, sys, os
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M
from data import build_records, per_user_folds
FIELDS=("user_id","domain","tool","data_type")
ep=int(os.environ.get("PERM_EPOCHS",200))

def _lg(p,eps=1e-3):
    p=np.clip(p,eps,1-eps); return np.log(p/(1-p))

def priors(tr, ev, alpha=5.0):
    glob=float(np.mean([r["label"] for r in tr]))
    ur=defaultdict(lambda:[0.,0.]); cr=defaultdict(lambda:[0.,0.])
    for r in tr:
        a=ur[r["user_id"]]; a[0]+=r["label"]; a[1]+=1
        b=cr[(r["tool"],r["data_type"])]; b[0]+=r["label"]; b[1]+=1
    out=[]
    for r in ev:
        a=ur.get(r["user_id"],[0.,0.]); b=cr.get((r["tool"],r["data_type"]),[0.,0.])
        pu=(a[0]+alpha*glob)/(a[1]+alpha); pc=(b[0]+alpha*glob)/(b[1]+alpha)
        out.append(0.5*(pu+pc))          # 先验概率(两个先验的均值)
    return np.array(out,dtype=np.float32)

def fm(tr,te,lam,dim=16,epochs=150,lr=0.05,wd=1e-6,seed=42):
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
    # 权重 = 1 + lam * |y - prior|   (先验错得越离谱, 权重越大)
    pr=priors(tr,tr)
    w=torch.tensor(1.0+lam*np.abs(np.array([r["label"] for r in tr])-pr),dtype=torch.float32)
    w=w/w.mean()
    def z(X):
        l=lin(X).sum(1).squeeze(-1)+b0; v=fac(X); s=v.sum(1)
        return l+0.5*((s*s).sum(-1)-(v*v).sum(-1).sum(-1))
    for _ in range(epochs):
        opt.zero_grad()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(z(Xtr),ytr,reduction='none')*w).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(z(enc(te)+offs)).numpy(), torch.sigmoid(z(Xtr)).numpy()

records,_=build_records(); folds=per_user_folds(records,5,42)
print(f"{'lambda (prior-surprise weight)':32s}{'Acc':>7}{'routine':>10}{'exception':>12}{'ratio':>8}")
for lam in (0.0, 1.0, 3.0, 10.0, 30.0):
    EXC,ROU,ACC=[],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        p,ptr=fm(tr,te,lam,epochs=ep)
        y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
        thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
        u=np.array([r["user_id"] for r in te])
        ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
        mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u])
        exc=(y!=mu); ov=(yh!=y)
        EXC+=list(ov[exc].astype(float)); ROU+=list(ov[~exc].astype(float)); ACC.append(100*(yh==y).mean())
    e,r_=100*np.mean(EXC),100*np.mean(ROU)
    tag=" <- baseline" if lam==0 else ""
    print(f"lam={lam:<28.0f}{np.mean(ACC):7.1f}{r_:9.1f}%{e:11.1f}%{e/max(r_,.1):7.1f}x{tag}")
