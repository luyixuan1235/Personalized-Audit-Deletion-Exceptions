"""Does per-user reweighting rescue CROWD-ALIGNED exceptions only?
Hypothesis (from the SPA flip): reweighting can cancel the habit prior but not the crowd
prior, because the crowd's parameters are shared. Prediction: in BOTH directions, weighting
helps exceptions with kappa(x)=y(x) and not those with kappa(x)!=y(x).
Output: results/table43_asym_kappa.json"""
import numpy as np, torch, sys, os, json, gc
sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import build_records, per_user_folds
from collections import defaultdict
FIELDS=("user_id","domain","tool","data_type")
EP=int(os.environ.get("PERM_EPOCHS",150))

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
cells=[{} for _ in range(4)]  # (dir,aligned) -> list, per mode handled below
out={}
for pw in (False,True):
    buckets=defaultdict(list)
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        p,ptr=fm(tr,te,pw)
        y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
        thr=float(M.equal_error_threshold(ytr,p_:=ptr)); yh=(p>=thr).astype(int)
        ur={}
        for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
        mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
        # crowd label per (tool,dtype) from TRAINING fold
        cg=defaultdict(lambda:[0,0])
        for r in tr:
            a=cg[(r["tool"],r["data_type"])]; a[0]+=r["label"]; a[1]+=1
        for i,r in enumerate(te):
            u=r["user_id"]
            if u not in mu or y[i]==mu[u]: continue          # exceptions only
            k=cg.get((r["tool"],r["data_type"]))
            if not k or k[1]==0: continue
            kap=1 if k[0]/k[1]>=0.5 else 0
            direction="refusal" if y[i]==0 else "consent"
            aligned="crowd-aligned" if kap==y[i] else "crowd-contradicting"
            buckets[(direction,aligned)].append(float(yh[i]!=y[i]))
        del p,ptr; gc.collect()
        print(f"pw={pw} fold {f+1}/5", flush=True)
    out["peruser" if pw else "none"]={f"{d}/{a}": dict(override=round(100*float(np.mean(v)),1), n=len(v))
        for (d,a),v in sorted(buckets.items())}
    print(json.dumps(out,indent=1), flush=True)
json.dump(out, open('../results/table43_asym_kappa.json','w'), indent=1)
print("saved table43_asym_kappa.json")
