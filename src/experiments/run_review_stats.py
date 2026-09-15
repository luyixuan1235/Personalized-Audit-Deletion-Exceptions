"""Reviewer-driven statistical hardening, all from existing Wu prediction files.
 E1 m_u estimation: test-fold vs train-only vs full-record       -> robustness
 E2 user-cluster bootstrap 95% CIs for headline EORs             -> clustering
 E3 within-user permutation null for the EOR/routine ratio      -> base-rate null
 E4 EOR ratio stratified by |user grant rate - 0.5|              -> Def-2 threshold
 E5 direct two-sided EORs + cluster-bootstrap CI of difference   -> symmetry
Writes results/table34_review_stats.json
"""
import json, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data import build_records

def load(path):
    d = json.load(open(path)); users,y,yh=[],[],[]
    if "cf_only" in d:                       # CF: raw scores + stored deployed threshold
        blob=d["cf_only"]; thr=blob["metrics"]["threshold"]
        for p in blob["test_predictions"]:
            users.append(p["userID"]); y.append(int(p["rating"]))
            yh.append(1 if p["prediction"]>=thr else 0)
    else:                                     # IC / IC+CF: labelled predictions
        for pid, blob in d.items():
            for pred, gt in zip(blob["predictions"], blob["ground_truth"]):
                pp = pred.get("permission", {})
                for dtype, gt_label in gt.get("answer", {}).items():
                    if dtype not in pp: continue
                    users.append(pid); y.append(1 if "Yes" in gt_label else 0)
                    yh.append(1 if "Yes" in pp[dtype].get("label","") else 0)
    return np.array(users), np.array(y), np.array(yh)

records,_ = build_records()
full = {}
for r in records: full.setdefault(r["user_id"], []).append(r["label"])
full_rate = {u: float(np.mean(v)) for u,v in full.items()}
full_n    = {u: len(v) for u,v in full.items()}

FILES = {"CF":"../wu-repo/results/cf_only_predictions.json",
         "IC":"../wu-repo/results/ic_only_predictions.json",
         "IC+CF":"../wu-repo/results/ic_cf_predictions.json"}

def rates(y,yh,mu):
    exc=(y!=mu); ov=(yh!=y)
    e=float(ov[exc].mean()) if exc.any() else np.nan
    r=float(ov[~exc].mean())
    return e,r,(e/r if r>0 else np.nan),int(exc.sum())

out={}
rng=np.random.default_rng(42)
for name,path in FILES.items():
    u,y,yh = load(path)
    res={}
    # per-user test grant sums for train-rate reconstruction
    uu=np.unique(u)
    tsum={x: int(y[u==x].sum()) for x in uu}; tn={x: int((u==x).sum()) for x in uu}
    def mu_of(kind):
        m={}
        for x in uu:
            if kind=="test":  rate=y[u==x].mean()
            elif kind=="full": rate=full_rate.get(x, y[u==x].mean())
            else: # train-only = full minus test items
                if x in full_rate and full_n[x]>tn[x]:
                    rate=(full_rate[x]*full_n[x]-tsum[x])/(full_n[x]-tn[x])
                else: rate=y[u==x].mean()
            m[x]=1 if rate>=0.5 else 0
        return np.array([m[x] for x in u]), m
    # E1: three definitions
    for kind in ("test","train","full"):
        mu,_=mu_of(kind)
        e,r,ratio,ne=rates(y,yh,mu)
        res[f"m_u={kind}"]=dict(exc=round(100*e,1),routine=round(100*r,1),ratio=round(ratio,2),n_exc=ne)
    mu,mud=mu_of("train")
    # E2: user-cluster bootstrap (train-m_u)
    boots=[]
    for _ in range(2000):
        pick=rng.choice(uu,size=len(uu),replace=True)
        idx=np.concatenate([np.where(u==x)[0] for x in pick])
        boots.append(rates(y[idx],yh[idx],mu[idx])[:3])
    B=np.array(boots)
    res["cluster_CI"]=dict(
        exc=[round(100*q,1) for q in np.nanpercentile(B[:,0],[2.5,97.5])],
        routine=[round(100*q,1) for q in np.nanpercentile(B[:,1],[2.5,97.5])],
        ratio=[round(q,2) for q in np.nanpercentile(B[:,2],[2.5,97.5])])
    # E3: within-user permutation null of the ratio
    obs=rates(y,yh,mu)[2]
    nulls=[]
    ov=(yh!=y)
    for _ in range(2000):
        ovp=ov.copy()
        for x in uu:
            ix=np.where(u==x)[0]; ovp[ix]=rng.permutation(ovp[ix])
        exc=(y!=mu)
        e=ovp[exc].mean(); r=ovp[~exc].mean()
        nulls.append(e/r if r>0 else np.nan)
    N=np.array(nulls)
    res["perm_null"]=dict(observed=round(obs,2),
        null_mean=round(float(np.nanmean(N)),2),
        null_95=[round(q,2) for q in np.nanpercentile(N,[2.5,97.5])],
        p=round(float(np.nanmean(N>=obs)),4))
    # E4: stratify by |train grant rate - 0.5|
    tr_rate={}
    for x in uu:
        if x in full_rate and full_n[x]>tn[x]:
            tr_rate[x]=(full_rate[x]*full_n[x]-tsum[x])/(full_n[x]-tn[x])
        else: tr_rate[x]=y[u==x].mean()
    dev=np.array([abs(tr_rate[x]-0.5) for x in u])
    strat={}
    for lo,hi,lab in ((0,0.1,"|r-.5|<=0.1"),(0.1,0.3,"0.1-0.3"),(0.3,0.5,"0.3-0.5")):
        m=(dev>lo-1e-9)&(dev<=hi)
        if m.sum()<50: continue
        e,r,ratio,ne=rates(y[m],yh[m],mu[m])
        strat[lab]=dict(exc=round(100*e,1),routine=round(100*r,1),ratio=round(ratio,2),n_exc=ne)
    res["margin_strata"]=strat
    # E5: direct two-sided EORs + bootstrap CI of the difference
    pr=(mu==1)&(y==0); rc=(mu==0)&(y==1); ov=(yh!=y)
    d_obs=float(ov[pr].mean()-ov[rc].mean())
    diffs=[]
    for _ in range(2000):
        pick=rng.choice(uu,size=len(uu),replace=True)
        idx=np.concatenate([np.where(u==x)[0] for x in pick])
        prb=(mu[idx]==1)&(y[idx]==0); rcb=(mu[idx]==0)&(y[idx]==1)
        if prb.any() and rcb.any():
            diffs.append(float(ov[idx][prb].mean()-ov[idx][rcb].mean()))
    D=np.array(diffs)
    res["symmetry"]=dict(
        perm_refusals=round(100*float(ov[pr].mean()),1), n_pr=int(pr.sum()),
        restr_consents=round(100*float(ov[rc].mean()),1), n_rc=int(rc.sum()),
        diff_CI=[round(100*q,1) for q in np.percentile(D,[2.5,97.5])])
    out[name]=res
    print(name, json.dumps(res, indent=1))
json.dump(out, open('../results/table34_review_stats.json','w'), indent=1)
print("saved table34_review_stats.json")
