"""Seventh repair (reviewer-suggested): empirical-Bayes partial pooling.
Prop 1 prices a fixed global lambda; the theory-motivated escape is to let the
shrinkage adapt per (user,domain) cell: offset each cell's prediction toward its own
train residual mean, shrunk by kappa = n/(n+n0) (James-Stein-style partial pooling).
Baseline: calibrated FM. Metric: permissive refusals + both-direction EOR (train-m_u).
Output: results/table35_eb_pooling.json
"""
import json, os, sys, gc
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines_ext as B, metrics as M
from data import build_records, per_user_folds

records,_=build_records(); folds=per_user_folds(records,5,42)
out={}
cells=[("user_domain", lambda r:(r["user_id"],r["domain"])),
       ("user_tool",   lambda r:(r["user_id"],r["tool"]))]
N0=[1.0,3.0,10.0]
pool={ (cn,n0):{"y":[],"p":[],"u":[],"mu":[]} for cn,_ in cells for n0 in N0 }
pool["base"]={"y":[],"p":[],"u":[],"mu":[]}
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); prtr=B.fm_predict(tr,tr,objective="bce")
    yte=np.asarray(pr["y"]); pte=np.asarray(pr["p_allow"])
    ytr=np.asarray(prtr["y"]); ptr=np.asarray(prtr["p_allow"])
    ur={}
    for r in tr: ur.setdefault(r["user_id"],[]).append(r["label"])
    mu={u:(1 if np.mean(v)>=0.5 else 0) for u,v in ur.items()}
    mte=np.array([mu.get(r["user_id"],1) for r in te])
    uu=np.array([r["user_id"] for r in te])
    pool["base"]["y"]+=list(yte); pool["base"]["p"]+=list(pte)
    pool["base"]["u"]+=list(uu);  pool["base"]["mu"]+=list(mte)
    for cn,key in cells:
        resid={}
        for i,r in enumerate(tr):
            resid.setdefault(key(r),[]).append(ytr[i]-ptr[i])
        for n0 in N0:
            off={k:(len(v)/(len(v)+n0))*float(np.mean(v)) for k,v in resid.items()}
            pte2=np.clip(pte+np.array([off.get(key(r),0.0) for r in te]),0,1)
            ptr2=np.clip(ptr+np.array([off.get(key(r),0.0) for r in tr]),0,1)
            pool[(cn,n0)]["y"]+=list(yte); pool[(cn,n0)]["p"]+=list(pte2)
            pool[(cn,n0)]["u"]+=list(uu);  pool[(cn,n0)]["mu"]+=list(mte)
            pool[(cn,n0)].setdefault("thr",[]).append(float(M.equal_error_threshold(ytr,ptr2)))
    pool["base"].setdefault("thr",[]).append(float(M.equal_error_threshold(ytr,ptr)))
    del pr,prtr; gc.collect()
    print(f"fold {f+1}/5 done", flush=True)

def report(d,label):
    y=np.array(d["y"]); p=np.array(d["p"]); mu=np.array(d["mu"])
    thr=float(np.mean(d["thr"])); yh=(p>=thr).astype(int); ov=(yh!=y)
    exc=(y!=mu); pr_=(mu==1)&(y==0); rc=(mu==0)&(y==1)
    r=dict(acc=round(100*float((yh==y).mean()),1),
           exc=round(100*float(ov[exc].mean()),1),
           routine=round(100*float(ov[~exc].mean()),1),
           perm_refusals=round(100*float(ov[pr_].mean()),1),
           restr_consents=round(100*float(ov[rc].mean()),1))
    print(f"{label:24s} {r}", flush=True)
    return r

res={"base":report(pool["base"],"baseline FM")}
for cn,_ in cells:
    for n0 in N0:
        res[f"{cn}_n0={n0}"]=report(pool[(cn,n0)],f"EB {cn} n0={n0}")
json.dump(res,open('../results/table35_eb_pooling.json','w'),indent=1)
print("saved table35_eb_pooling.json")
