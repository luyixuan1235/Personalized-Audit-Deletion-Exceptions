"""Reviewer diagnostic: is the asymmetric remedy an artifact of the 69% global grant rate?

Per-user reweighting rescues restrictive users' rare consents (48.1->29.6%) but not
permissive users' rare refusals (30.5->31.3%). Hypothesis (reviewer): the pooled gradient
is dominated by the grant-majority current even after per-user balancing. Test: subsample
the TRAINING grants to a 50/50 global base rate and re-run the comparison. If the global
prior drives the asymmetry, balancing should let per-user weighting help refusals too.
"""
import numpy as np, torch, sys, os
from collections import defaultdict
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/experiments')
import _common as C, metrics as M
from data import build_records, per_user_folds
FIELDS=("user_id","domain","tool","data_type")
ep=15 if C.QUICK else 150

def fm(tr,te,peruser_w,dim=16,epochs=ep,lr=0.05,wd=1e-6,seed=42):
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
    if peruser_w:
        cnt=defaultdict(lambda:[0,0])
        for r in tr: cnt[r["user_id"]][int(r["label"])]+=1
        w=torch.tensor([max(cnt[r["user_id"]])/max(cnt[r["user_id"]][int(r["label"])],1) for r in tr],dtype=torch.float32)
        w=w/w.mean()
    else: w=torch.ones(len(tr))
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
rng=np.random.default_rng(0)
print(f"{'condition':40s}{'perm.refusals':>14}{'restr.consents':>15}")
for base,label in (('natural','natural base rate (0.69)'),('balanced','balanced base rate (0.50)')):
    for pw in (False,True):
        PR,RC=[],[]
        for f in range(5):
            tr=[records[i] for i in range(len(records)) if folds[i]!=f]
            te=[records[i] for i in range(len(records)) if folds[i]==f]
            if base=='balanced':
                g=[r for r in tr if r["label"]==1]; d=[r for r in tr if r["label"]==0]
                keep=rng.permutation(len(g))[:len(d)]
                tr=[g[i] for i in keep]+d
            p,ptr=fm(tr,te,pw)
            y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
            thr=float(M.equal_error_threshold(ytr,p_tr:=ptr)); yh=(p>=thr).astype(int)
            u=np.array([r["user_id"] for r in te])
            ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
            mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u])
            exc=(y!=mu); ov=(yh!=y)
            pr=exc&(y==0); rc=exc&(y==1)
            PR+=list(ov[pr].astype(float)); RC+=list(ov[rc].astype(float))
        print(f"{label+(' + per-user weight' if pw else ''):40s}{100*np.mean(PR):13.1f}%{100*np.mean(RC):14.1f}%")
