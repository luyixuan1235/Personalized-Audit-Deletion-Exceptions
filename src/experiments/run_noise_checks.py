"""Reviewer round 5: (A) are exceptions stable preferences or response noise?
(B) macro-user EOR; (C) is LLM self-reported confidence calibrated (ECE)?
Output: results/table37_noise_checks.json"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records

out={}
# ---------- A: within-user exception direction consistency ----------
records,_=build_records()
byu={}
for r in records: byu.setdefault(r["user_id"],[]).append(r)
pairs_same=pairs_tot=0
rng=np.random.default_rng(0)
null_frac=[]
excs=[]
for u,rs in byu.items():
    y=np.array([r["label"] for r in rs]); mu=1 if y.mean()>=0.5 else 0
    ex=[r for r in rs if r["label"]!=mu]
    excs.append(len(ex))
    # within same domain: do two exceptions by the same user agree in direction?
    # (direction is forced equal by construction of mu; instead test CONTEXT consistency:
    #  does an exception recur on the same (tool,dtype) conjunction or domain?)
for u,rs in byu.items():
    y=np.array([r["label"] for r in rs]); mu=1 if y.mean()>=0.5 else 0
    ex=[r for r in rs if r["label"]!=mu]
    if len(ex)<1: continue
    keys_ex=set((r["tool"],r["data_type"]) for r in ex)
    dom_ex=set(r["domain"] for r in ex)
    # recurrence: fraction of exceptions sharing (tool,dtype) or domain with ANOTHER exception
    for i,r in enumerate(ex):
        others=[e for j,e in enumerate(ex) if j!=i]
        if not others: continue
        pairs_tot+=1
        if any((r["tool"]==o["tool"] and r["data_type"]==o["data_type"]) or r["domain"]==o["domain"] for o in others):
            pairs_same+=1
    # null: pick the same number of random decisions from this user, same test
    for _ in range(5):
        pick=rng.choice(len(rs),size=len(ex),replace=False)
        sub=[rs[i] for i in pick]
        loc=0; tot=0
        for i,r in enumerate(sub):
            others=[e for j,e in enumerate(sub) if j!=i]
            if not others: continue
            tot+=1
            if any((r["tool"]==o["tool"] and r["data_type"]==o["data_type"]) or r["domain"]==o["domain"] for o in others):
                loc+=1
        if tot: null_frac.append(loc/tot)
out["A_exception_recurrence"]=dict(
    observed=round(pairs_same/pairs_tot,3),
    null_random_decisions=round(float(np.mean(null_frac)),3),
    n_exceptions_with_peer=pairs_tot,
    note="fraction of a user's exceptions sharing domain or (tool,dtype) with another exception by the same user; null = same-size random subsets of the user's decisions")

# ---------- B: macro-user EOR ----------
def load(path):
    d = json.load(open(path)); users,y,yh,cf=[],[],[],[]
    if "cf_only" in d:
        blob=d["cf_only"]; thr=blob["metrics"]["threshold"]
        for p in blob["test_predictions"]:
            users.append(p["userID"]); y.append(int(p["rating"]))
            yh.append(1 if p["prediction"]>=thr else 0); cf.append(np.nan)
    else:
        for pid, blob in d.items():
            for pred, gt in zip(blob["predictions"], blob["ground_truth"]):
                pp = pred.get("permission", {})
                for dtype, gt_label in gt.get("answer", {}).items():
                    if dtype not in pp: continue
                    users.append(pid); y.append(1 if "Yes" in gt_label else 0)
                    yh.append(1 if "Yes" in pp[dtype].get("label","") else 0)
                    cf.append(float(pp[dtype].get("score", np.nan)))
    return np.array(users), np.array(y), np.array(yh), np.array(cf)

FILES={"CF":"../wu-repo/results/cf_only_predictions.json",
       "IC":"../wu-repo/results/ic_only_predictions.json",
       "IC+CF":"../wu-repo/results/ic_cf_predictions.json"}
macro={}
for name,path in FILES.items():
    u,y,yh,cf=load(path)
    gr={x:y[u==x].mean() for x in np.unique(u)}
    mu=np.array([1 if gr[x]>=0.5 else 0 for x in u]); exc=(y!=mu); ov=(yh!=y)
    per_e=[]; per_r=[]
    for x in np.unique(u):
        m=(u==x)
        if (m&exc).any(): per_e.append(ov[m&exc].mean())
        if (m&~exc).any(): per_r.append(ov[m&~exc].mean())
    macro[name]=dict(macro_exc=round(100*float(np.mean(per_e)),1),
                     macro_routine=round(100*float(np.mean(per_r)),1),
                     macro_ratio=round(float(np.mean(per_e)/np.mean(per_r)),2),
                     n_users_with_exc=len(per_e))
out["B_macro_user_EOR"]=macro

# ---------- C: calibration of self-reported confidence ----------
def ece(conf, correct, bins=10):
    edges=np.linspace(0.5,1.0,bins+1); tot=len(conf); e=0.0; rel=[]
    for i in range(bins):
        m=(conf>=edges[i])&(conf<edges[i+1]+(1e-9 if i==bins-1 else 0))
        if m.sum()==0: continue
        acc=correct[m].mean(); c=conf[m].mean()
        e+=m.sum()/tot*abs(acc-c); rel.append((round(float(c),3),round(float(acc),3),int(m.sum())))
    return e, rel
cal={}
for name in ("IC","IC+CF"):
    u,y,yh,cf=load(FILES[name])
    ok=~np.isnan(cf); correct=(yh==y)[ok].astype(float); c=cf[ok]
    gr={x:y[u==x].mean() for x in np.unique(u)}
    mu=np.array([1 if gr[x]>=0.5 else 0 for x in u]); exc=(y!=mu)[ok]
    e_all,rel=ece(c,correct)
    e_exc,_=ece(c[exc],correct[exc])
    cal[name]=dict(ECE_all=round(float(e_all),3), ECE_exceptions=round(float(e_exc),3),
                   mean_conf=round(float(c.mean()),3), acc=round(float(correct.mean()),3),
                   mean_conf_exc=round(float(c[exc].mean()),3), acc_exc=round(float(correct[exc].mean()),3))
out["C_confidence_calibration"]=cal
json.dump(out,open('../results/table37_noise_checks.json','w'),indent=1)
print(json.dumps(out,indent=1))
