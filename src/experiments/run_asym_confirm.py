"""Confirming intervention for the asymmetry carrier (user-majority composition).
Both arms hold the marginal label rate at 0.5; only the composition varies.
Arm A (natural composition, ~71% grant-majority users) exists as table31's balanced rows.
Arm B: subsample grant-majority users until deny-majority users dominate (~70%), then
rebalance labels to 0.5. Prediction: the rescue direction flips to refusals.
Output: results/table44_asym_confirm.json"""
import numpy as np, torch, sys, os, json, gc
sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
from data import build_records, per_user_folds
from collections import defaultdict
FIELDS=("user_id","domain","tool","data_type")

def fm(tr,te,peruser,dim=16,epochs=150,lr=0.05,wd=1e-6,seed=42):
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

records,_=build_records()
rng=np.random.default_rng(7)
# flip the composition: keep all deny-majority users, subsample grant-majority users
byu={}
for r in records: byu.setdefault(r["user_id"],[]).append(r["label"])
gmaj=[u for u,v in byu.items() if np.mean(v)>=0.5]; dmaj=[u for u,v in byu.items() if np.mean(v)<0.5]
keep_g=list(rng.permutation(gmaj)[:max(1,int(len(dmaj)*0.43))])   # ~30/70 composition
keep=set(keep_g)|set(dmaj)
sub=[r for r in records if r["user_id"] in keep]
comp=np.mean([1 if np.mean(byu[u])>=0.5 else 0 for u in keep])
print(f"arm B: users={len(keep)} grant-majority share={100*comp:.1f}% records={len(sub)} marginal={np.mean([r['label'] for r in sub]):.2f}", flush=True)
folds=per_user_folds(sub,5,42)
out={"composition_grant_majority_share":round(100*float(comp),1)}
for pw in (False,True):
    PR,RC=[],[]
    for f in range(5):
        tr=[sub[i] for i in range(len(sub)) if folds[i]!=f]
        te=[sub[i] for i in range(len(sub)) if folds[i]==f]
        # rebalance marginal to 0.5 within the flipped composition
        g=[r for r in tr if r["label"]==1]; d=[r for r in tr if r["label"]==0]
        if len(g)>len(d):
            pick=rng.permutation(len(g))[:len(d)]; tr=[g[i] for i in pick]+d
        elif len(d)>len(g):
            pick=rng.permutation(len(d))[:len(g)]; tr=g+[d[i] for i in pick]
        p,ptr=fm(tr,te,pw)
        y=np.array([r["label"] for r in te]); ytr=np.array([r["label"] for r in tr])
        thr=float(M.equal_error_threshold(ytr,ptr)); yh=(p>=thr).astype(int)
        ur={}
        for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
        mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
        m=np.array([mu.get(r["user_id"],0) for r in te])
        ov=(yh!=y); pr_=(m==1)&(y==0); rc=(m==0)&(y==1)
        PR+=list(ov[pr_].astype(float)); RC+=list(ov[rc].astype(float))
        print(f"pw={pw} fold {f+1}/5", flush=True)
        del p,ptr; gc.collect()
    out["peruser" if pw else "none"]=dict(
        perm_refusals=round(100*float(np.mean(PR)),1), n_pr=len(PR),
        restr_consents=round(100*float(np.mean(RC)),1), n_rc=len(RC))
    print(out, flush=True)
json.dump(out, open('../results/table44_asym_confirm.json','w'), indent=1)
print("saved table44_asym_confirm.json")
