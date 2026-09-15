"""Multi-seed replication of the history-inversion intervention: repeat the flip
with different random user sets and folds. Output: results/table69_flip_seeds.json"""
import json, os, sys
import numpy as np
sys.path.insert(0,'.')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import baselines_ext as B
from data import build_records, per_user_folds

records,_=build_records(); folds=per_user_folds(records,5,42)
users=sorted({r["user_id"] for r in records})
out={}
for seed,f in ((42,0),(7,1),(202,2),(99,3),(1234,4)):
    rng=np.random.default_rng(seed)
    flip=set(rng.choice(users,size=20,replace=False))
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    tr_flip=[dict(r,label=1-r["label"]) if r["user_id"] in flip else r for r in tr]
    pb=B.fm_predict(tr,te,objective="bce"); pf=B.fm_predict(tr_flip,te,objective="bce")
    p0=np.asarray(pb["p_allow"]); p1=np.asarray(pf["p_allow"])
    u=np.array([r["user_id"] for r in te]); mf=np.isin(u,list(flip))
    gr={}
    for r in tr: gr.setdefault(r["user_id"],[]).append(r["label"])
    followed=0; counted=0
    for uu in flip:
        mm=(u==uu)
        if mm.sum()==0: continue
        counted+=1
        m0=1 if np.mean(gr[uu])>=0.5 else 0
        if (1 if (p1[mm]>=0.5).mean()>=0.5 else 0)==1-m0: followed+=1
    out[f"seed={seed},fold={f}"]=dict(
        p_before=round(float(p0[mf].mean()),3), p_after=round(float(p1[mf].mean()),3),
        control_abs_shift=round(float(np.abs(p1[~mf]-p0[~mf]).mean()),3),
        follows=f"{followed}/{counted}")
    print(f"seed={seed} fold={f}:", out[f"seed={seed},fold={f}"], flush=True)
rd=os.environ.get("PERM_RESULTS_DIR","../results")
json.dump(out, open(os.path.join(rd,"table69_flip_seeds.json"),"w"), indent=1)
print("saved table69_flip_seeds.json")
