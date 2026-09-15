"""最终方法: 用例外规则做【弃权信号】, 而不是覆盖预测.
'这个用户以前在这个 domain 上破过例 -> 别自动决策, 去问他.'
对照: (a) 不弃权 (b) 按模型置信度弃权 (领域现行做法) (c) 按例外历史弃权 (我们的)"""
import numpy as np, sys, gc
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M, baselines_ext as B
from data import build_records, per_user_folds
ATTRS=("domain","tool","data_type")
records,_=build_records(); folds=per_user_folds(records,5,42)

cache=[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    pr=B.fm_predict(tr,te,objective="bce"); prtr=B.fm_predict(tr,tr,objective="bce")
    thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
    cache.append((tr,te,np.asarray(pr["y"]),np.asarray(pr["p_allow"]),thr))
    del pr,prtr; gc.collect()

def exc_flags(tr, te, sup, pur):
    """该用户是否在此请求的某个属性上有过例外历史 -> 弃权"""
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
    rules={k for k,(ne,n) in cnt.items() if n>=sup and ne/n>=pur}
    return np.array([any((r["user_id"],a,r[a]) in rules for a in ATTRS) for r in te])

def report(label, defer_fn):
    COV,EXC,ROU,FG=[],[],[],[]
    for tr,te,y,p,thr in cache:
        yh=(p>=thr).astype(int)
        d=defer_fn(tr,te,y,p,thr)             # True = 弃权(去问用户)
        auto=~d
        u_=np.array([r["user_id"] for r in te])
        urate={uu: float(y[u_==uu].mean()) for uu in np.unique(u_)}
        m_=np.array([1 if urate[uu]>=0.5 else 0 for uu in u_])
        exc=(y!=m_); ov=(yh!=y)
        COV.append(100*auto.mean())
        if (auto&exc).sum(): EXC+=list(ov[auto&exc].astype(float))
        if (auto&~exc).sum(): ROU+=list(ov[auto&~exc].astype(float))
        dn=auto&(y==0)
        if dn.sum(): FG+=list(yh[dn].astype(float))
    print(f"{label:38s}{np.mean(COV):8.1f}%{100*np.mean(ROU):10.1f}%{100*np.mean(EXC):12.1f}%{100*np.mean(FG):11.1f}%", flush=True)
    _OUT[label]=dict(coverage=float(np.mean(COV)),routine=float(100*np.mean(ROU)),exception=float(100*np.mean(EXC)),fgr=float(100*np.mean(FG)))

_OUT={}
print("在【自动决策】的那部分请求上衡量 (弃权的去问用户)\n")
print(f"{'弃权策略':38s}{'自动决策覆盖':>9}{'routine覆盖':>11}{'例外覆盖':>13}{'误授权':>12}")
report("无弃权 (现状)", lambda tr,te,y,p,thr: np.zeros(len(te),bool))
for q in (0.25,0.40):
    report(f"按模型置信度弃权 (最低{int(q*100)}%)",
           lambda tr,te,y,p,thr,q=q: np.abs(p-thr)<=np.quantile(np.abs(p-thr),q))
for sup,pur in ((2,0.5),(2,0.34),(1,0.5)):
    f=lambda tr,te,y,p,thr,s=sup,pu=pur: exc_flags(tr,te,s,pu)
    cov=np.mean([100*(~exc_flags(tr,te,sup,pur)).mean() for tr,te,_,_,_ in cache])
    report(f"★ 按例外历史弃权 (sup>={sup},pur>={pur})", f)

import json as _json, os as _os
_json.dump(_OUT, open(_os.path.join(_os.environ.get("PERM_RESULTS_DIR","../results"),"table51_deferral_exc.json"),"w"), indent=1)
print("saved table51_deferral_exc.json")
