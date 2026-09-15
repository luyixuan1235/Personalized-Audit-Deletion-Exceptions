"""W5: does the asymmetric remedy track the corpus majority? Wu (grant-majority):
per-user weight helps restrictive consents only. SPA (deny-majority): flip test.
Run: PERM_DATA_DIR=../data_spa PERM_RESULTS_DIR=../results_spa python3 experiments/run_asym_spa.py
Output: results_spa/table41_asym_spa.json"""
import numpy as np, torch, sys, os, json, gc
sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import build_records, per_user_folds
from collections import defaultdict
FIELDS=("user_id","domain","tool","data_type")
EP=int(os.environ.get("PERM_EPOCHS", 100))

def fm(tr,te,peruser,dim=16,epochs=EP,lr=0.05,wd=1e-6,seed=42):
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
    if peruser:
        cnt=defaultdict(lambda:[0,0])
        for r in tr: cnt[r["user_id"]][int(r["label"])]+=1
        w=torch.tensor([max(cnt[r["user_id"]])/max(cnt[r["user_id"]][int(r["label"])],1) for r in tr],dtype=torch.float32)
        w=w/w.mean()
    else: w=torch.ones(len(tr))
    def logits(X):
        l=lin(X).sum(1).squeeze(-1)+b0; v=fac(X); s=v.sum(1)
        return l+0.5*((s*s).sum(-1)-(v*v).sum(-1).sum(-1))
    for _ in range(epochs):
        opt.zero_grad()
        loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits(Xtr),ytr,reduction='none')*w).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(logits(enc(te)+offs)).numpy(), torch.sigmoid(logits(Xtr)).numpy()

records,_=build_records(); folds=per_user_folds(records,5,42)
print(f"records={len(records)}",flush=True)
out={}
for pw in (False,True):
    PR,RC,ACC=[],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        p,ptr=fm(tr,te,pw)
        y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
        thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
        ur={}
        for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
        mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
        m=np.array([mu.get(r["user_id"],0) for r in te])
        ov=(yh!=y); pr_=(m==1)&(y==0); rc=(m==0)&(y==1)
        PR+=list(ov[pr_].astype(float)); RC+=list(ov[rc].astype(float)); ACC.append(float((yh==y).mean()))
        print(f"pw={pw} fold {f+1}/5 done", flush=True)
        del p,ptr; gc.collect()
    out["peruser" if pw else "none"]=dict(acc=round(100*float(np.mean(ACC)),1),
        perm_refusals=round(100*float(np.mean(PR)),1), n_pr=len(PR),
        restr_consents=round(100*float(np.mean(RC)),1), n_rc=len(RC))
    print(out, flush=True)
rd=os.environ.get("PERM_RESULTS_DIR","../results")
json.dump(out, open(os.path.join(rd,"table41_asym_spa.json"),"w"), indent=1)
print("saved table41_asym_spa.json")
