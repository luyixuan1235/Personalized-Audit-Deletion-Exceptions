"""Exception-aware: per-user 例外规则. FM 只训一次, 缓存后套不同规则配置."""
import numpy as np, sys, os, gc
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M, baselines_ext as B
from data import build_records, per_user_folds
ATTRS=("domain","tool","data_type")
records,_=build_records(); folds=per_user_folds(records,5,42)

# ---- 只训一次 FM, 缓存每折的预测
cache=[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); prtr=B.fm_predict(tr,tr,objective="bce")
    thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
    cache.append((tr,te,np.asarray(pr["y"]),np.asarray(pr["p_allow"]),thr))
    del pr,prtr; gc.collect()
print("FM 训练完成, 开始套规则\n")

def mine(tr, sup, pur):
    ur=defaultdict(lambda:[0,0])
    for r in tr:
        a=ur[r["user_id"]]; a[0]+=r["label"]; a[1]+=1
    mu={u:(1 if a[0]/max(a[1],1)>=0.5 else 0) for u,a in ur.items()}
    cnt=defaultdict(lambda:[0,0])
    for r in tr:
        u=r["user_id"]
        if u not in mu: continue
        for a in ATTRS:
            k=(u,a,r[a]); cnt[k][1]+=1
            if r["label"]!=mu[u]: cnt[k][0]+=1
    rules={k:1-mu[k[0]] for k,(ne,n) in cnt.items() if n>=sup and ne/n>=pur}
    return rules, mu

def run(sup,pur,label):
    EXC,ROU,ACC,FIRE,CORR=[],[],[],[],[]
    for tr,te,y,p,thr in cache:
        yh=(p>=thr).astype(int)
        if sup>0:
            rules,mu=mine(tr,sup,pur); fired=corr=0
            for i,r in enumerate(te):
                u=r["user_id"]
                if u not in mu: continue
                for a in ATTRS:
                    if (u,a,r[a]) in rules:
                        new=rules[(u,a,r[a])]
                        if new!=yh[i]:
                            fired+=1
                            if new==y[i]: corr+=1
                            else: corr-=1
                        yh[i]=new; break
            FIRE.append(100*fired/len(te)); CORR.append(corr)
        u_=np.array([r["user_id"] for r in te])
        urate={uu: float(y[u_==uu].mean()) for uu in np.unique(u_)}
        m_=np.array([1 if urate[uu]>=0.5 else 0 for uu in u_])
        exc=(y!=m_); ov=(yh!=y)
        EXC+=list(ov[exc].astype(float)); ROU+=list(ov[~exc].astype(float)); ACC.append(100*(yh==y).mean())
    e,r_=100*np.mean(EXC),100*np.mean(ROU)
    fr=np.mean(FIRE) if FIRE else 0.
    print(f"{label:32s}{np.mean(ACC):7.1f}{r_:9.1f}%{e:11.1f}%{e/max(r_,.1):7.1f}x{fr:9.1f}%")
    return e

print(f"{'method':32s}{'Acc':>7}{'routine':>10}{'exception':>12}{'ratio':>8}{'规则改判':>10}")
base=run(0,0,"base FM (calibrated)")
best=(base,None)
for sup,pur in ((2,0.5),(2,0.67),(2,0.8),(3,0.5),(3,0.67),(4,0.6)):
    e=run(sup,pur,f"+ rules(sup>={sup}, purity>={pur})")
    if e<best[0]: best=(e,(sup,pur))
print(f"\nbaseline exception override = {base:.1f}%")
print(f"best = {best[0]:.1f}%  {best[1]}")
d=base-best[0]
print(f"-> {'✓ 例外覆盖率降低 %.1f 点' % d if d>1 else '✗ 无实质改善'}")
