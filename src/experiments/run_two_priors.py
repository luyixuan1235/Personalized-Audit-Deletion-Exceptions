"""核心图: user majority x crowd majority -> override rate
每个决策问两件事: 它是否违背用户自己的多数? 是否违背群体对该请求类型的多数?"""
import json, numpy as np, sys
from collections import defaultdict
sys.path.insert(0,'.')
from analyze_wu_peruser import load_cf, load_llm

def keys_for(which):
    if which=='cf':
        d=json.load(open('../results/cf_only_predictions.json'))['cf_only']['test_predictions']
        return [r['itemID'] for r in d]
    fn='../results/ic_only_predictions.json' if which=='ic' else '../results/ic_cf_predictions.json'
    d=json.load(open(fn)); ks=[]
    for pid,dat in d.items():
        if not isinstance(dat,dict) or dat.get('skipped') or 'error' in dat: continue
        preds={p.get('id'):p.get('permission',{}) for p in dat.get('predictions',[])}
        for gt in dat.get('ground_truth',[]):
            pp=preds.get(gt['id'])
            if not pp: continue
            for dt in gt.get('answer',{}):
                if dt in pp: ks.append(dt)
    return ks

SRC=[('CF only (LightGCN+BPR)','cf', lambda: load_cf('../results/cf_only_predictions.json')[:3]),
     ('IC only (LLM)','ic',          lambda: load_llm('../results/ic_only_predictions.json')[:3]),
     ('IC+CF (deployed, 85.1%)','iccf', lambda: load_llm('../results/ic_cf_predictions.json')[:3])]

def _wilson(k,n,z=1.96):
    if n==0: return (float('nan'),float('nan'))
    ph=k/n; d=1+z*z/n; c=ph+z*z/(2*n); h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))
    return (100*(c-h)/d, 100*(c+h)/d)
_OUT={}
for name,which,f in SRC:
    u,y,yh=f(); k=np.array(keys_for(which)[:len(y)])
    if len(k)!=len(y): print(f"skip {name}"); continue
    # 用户多数 / 群体多数 (均留一)
    um={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
    cg=defaultdict(lambda:[0,0])
    for kk,yy in zip(k,y): a=cg[kk]; a[0]+=int(yy); a[1]+=1
    UM=np.array([1 if um[uu]>=0.5 else 0 for uu in u])                       # 用户多数标签
    CM=np.array([1 if (cg[kk][0]-yy)/max(cg[kk][1]-1,1)>=0.5 else 0
                 for kk,yy in zip(k,y)])                                     # 群体多数标签(留一)
    ov=(yh!=y).astype(float)                                                 # 该决策被覆盖
    print(f"\n### {name}")
    print(f"    该决策被模型覆盖的比例（行=是否违背用户多数, 列=是否违背群体多数）\n")
    print(f"    {'':22}{'与群体一致':>14}{'违背群体':>14}")
    _OUT[name]={}
    for agree_u in (True,False):
        row=f"    {'与用户多数一致' if agree_u else '违背用户多数(例外)':<22}"
        for agree_c in (True,False):
            m=((y==UM)==agree_u)&((y==CM)==agree_c)
            row+= f"{100*ov[m].mean():11.1f}% (n={m.sum():4d})" if m.sum()>5 else f"{'-':>14}"
            if m.sum()>5:
                cell=('agree_habit' if agree_u else 'contra_habit')+'/'+('agree_crowd' if agree_c else 'contra_crowd')
                nerr=int(ov[m].sum()); ntot=int(m.sum())
                _OUT[name][cell]=dict(override=float(100*ov[m].mean()), n=ntot,
                                      wilson95=list(_wilson(nerr,ntot)))
        print(row)

import json as _json, os as _os
_json.dump(_OUT, open(_os.path.join(_os.environ.get("PERM_RESULTS_DIR","../results"),"table52_two_priors.json"),"w"), indent=1)
print("saved table52_two_priors.json (with Wilson 95% CIs)")
