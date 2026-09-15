"""#2c: do habit and crowd effects survive controls? Logistic regression of
override on contra-habit + contra-crowd + log support + data-type fixed effects,
on our FM (bce) predictions, Wu corpus. Output: results/table57_regression.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import metrics as M
import baselines_ext as B
from data import build_records, per_user_folds
from collections import defaultdict

records,_=build_records(); folds=per_user_folds(records,5,42)
conj=lambda r:(r["tool"],r["data_type"])
rows=[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); ptr=B.fm_predict(tr,tr,objective="bce")
    thr=float(M.equal_error_threshold(np.asarray(ptr["y"]),np.asarray(ptr["p_allow"])))
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"]); yh=(p>=thr).astype(int)
    ur=defaultdict(list)
    for r in tr: ur[r["user_id"]].append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    cg=defaultdict(lambda:[0,0]); sup=defaultdict(int)
    for r in tr:
        a=cg[conj(r)]; a[0]+=r["label"]; a[1]+=1
        sup[(r["user_id"],)+conj(r)]+=1
    for i,r in enumerate(te):
        u=r["user_id"]
        if u not in mu: continue
        c=cg[conj(r)]
        if c[1]==0: continue
        cm=1 if c[0]/c[1]>=0.5 else 0
        rows.append(dict(ov=int(yh[i]!=y[i]),
                         ch=int(y[i]!=mu[u]), cc=int(y[i]!=cm),
                         lsup=float(np.log1p(sup[(u,)+conj(r)])),
                         lhist=float(np.log(len(ur[u]))),
                         novel=int(sup[(u,)+conj(r)]==0),
                         dtype=r["data_type"]))
# logistic regression with data-type fixed effects (torch, full batch)
import torch
dts=sorted({r["dtype"] for r in rows}); di={d:i for i,d in enumerate(dts)}
X=torch.tensor([[r["ch"],r["cc"],r["ch"]*r["cc"],r["lsup"],r["lhist"],r["novel"]] for r in rows],dtype=torch.float32)
FE=torch.tensor([di[r["dtype"]] for r in rows])
Y=torch.tensor([float(r["ov"]) for r in rows])
torch.manual_seed(42)
w=torch.zeros(6,requires_grad=True); fe=torch.zeros(len(dts),requires_grad=True); b0=torch.zeros(1,requires_grad=True)
opt=torch.optim.LBFGS([w,fe,b0],max_iter=200)
def closure():
    opt.zero_grad()
    z=X@w+fe[FE]+b0
    l=torch.nn.functional.binary_cross_entropy_with_logits(z,Y)+1e-4*(fe*fe).sum()
    l.backward(); return l
opt.step(closure)
# cluster (user? no user id kept... use dtype-cluster? better: recompute with user bootstrap)
# simple nonparametric: refit on 200 bootstrap resamples of rows clustered by nothing -> report naive SE via Hessian approx: skip; report coefficients + resample by decision
def fit(idx):
    Xs,FEs,Ys=X[idx],FE[idx],Y[idx]
    ww=torch.zeros(6,requires_grad=True); ff=torch.zeros(len(dts),requires_grad=True); bb=torch.zeros(1,requires_grad=True)
    o=torch.optim.LBFGS([ww,ff,bb],max_iter=100)
    def cl():
        o.zero_grad()
        l=torch.nn.functional.binary_cross_entropy_with_logits(Xs@ww+ff[FEs]+bb,Ys)+1e-4*(ff*ff).sum()
        l.backward(); return l
    o.step(cl); return ww.detach().numpy()
rng=np.random.default_rng(0); boots=[]
for _ in range(100):
    idx=torch.tensor(rng.integers(0,len(rows),len(rows)))
    boots.append(fit(idx))
boots=np.array(boots)
names=["contra_habit","contra_crowd","interaction","log_support","log_histlen","novel_conj"]
out={n:{ "coef":float(w.detach().numpy()[i]),
        "ci95":[float(np.percentile(boots[:,i],2.5)),float(np.percentile(boots[:,i],97.5))]}
     for i,n in enumerate(names)}
out["n"]=len(rows); out["note"]="logit(override) ~ contra_habit + contra_crowd + interaction + log1p(support) + log(user history length) + first-of-kind flag + data-type FE; FM/bce, Wu, 5-fold; decision bootstrap 100x"
print(json.dumps(out,indent=1))
json.dump(out, open(os.path.join(os.environ.get("PERM_RESULTS_DIR","../results"),"table57b_regression_ext.json"),"w"), indent=1)
print("saved table57_regression.json")
