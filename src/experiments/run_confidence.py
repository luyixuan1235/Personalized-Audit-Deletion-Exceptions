"""例外是否被挤出高置信区域? 模型对'例外'的置信度分布 vs 对'常规'的."""
import json, numpy as np, sys
from collections import defaultdict
sys.path.insert(0,'.')
from analyze_wu_peruser import load_cf, load_llm

# --- CF: 用到决策边界的距离作为置信度
d=json.load(open('../results/cf_only_predictions.json'))['cf_only']
rec=d['test_predictions']; thr=d['metrics']['threshold']
y=np.array([1 if r['rating']>0 else 0 for r in rec])
s=np.array([r['prediction'] for r in rec],float)
u=np.array([r['userID'] for r in rec])
um={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
UM=np.array([1 if um[uu]>=0.5 else 0 for uu in u])
conf=np.abs(s-thr); conf=conf/conf.max()
yh=(s>=thr).astype(int); ov=(yh!=y)
exc=(y!=UM)   # 例外 = 违背自己的多数

print("### CF (LightGCN+BPR)  —— 置信度 = |score - threshold|, 归一化")
print(f"    {'置信度分位':>12}{'常规决策 被覆盖':>18}{'例外决策 被覆盖':>18}{'例外占比':>10}")
qs=np.percentile(conf,[25,50,75])
for lo,hi,lab in ((0,qs[0],'最低25%'),(qs[0],qs[1],'25-50%'),(qs[1],qs[2],'50-75%'),(qs[2],9,'最高25%')):
    m=(conf>=lo)&(conf<hi)
    a=100*ov[m&~exc].mean() if (m&~exc).sum()>5 else float('nan')
    b=100*ov[m&exc].mean() if (m&exc).sum()>5 else float('nan')
    print(f"    {lab:>12}{a:14.1f}% (n={(m&~exc).sum():4d}){b:11.1f}% (n={(m&exc).sum():3d}){100*exc[m].mean():9.1f}%")
print(f"\n    例外决策的平均置信度 = {conf[exc].mean():.3f}   常规 = {conf[~exc].mean():.3f}")
print(f"    -> 模型对'例外'的置信度{'更低(可以deferral)' if conf[exc].mean()<conf[~exc].mean() else '并不更低 —— 例外被自信地覆盖'}")

# --- LLM: 用它自己输出的 confidence score
for fn,lab in (('../results/ic_only_predictions.json','IC only (LLM)'),
               ('../results/ic_cf_predictions.json','IC+CF (deployed)')):
    D=json.load(open(fn)); U,Y,YH,CF=[],[],[],[]
    for pid,dat in D.items():
        if not isinstance(dat,dict) or dat.get('skipped') or 'error' in dat: continue
        preds={p.get('id'):p.get('permission',{}) for p in dat.get('predictions',[])}
        for gt in dat.get('ground_truth',[]):
            pp=preds.get(gt['id'])
            if not pp: continue
            for dt,gl in gt.get('answer',{}).items():
                if dt not in pp: continue
                U.append(pid); Y.append(1 if 'Yes' in gl else 0)
                YH.append(1 if 'Yes' in pp[dt].get('label','') else 0)
                CF.append(float(pp[dt].get('score',0.5)))
    U,Y,YH,CF=np.array(U),np.array(Y),np.array(YH),np.array(CF)
    um={uu: float(Y[U==uu].mean()) for uu in np.unique(U)}
    UM=np.array([1 if um[uu]>=0.5 else 0 for uu in U])
    exc=(Y!=UM); ov=(YH!=Y)
    print(f"\n### {lab}  —— 置信度 = LLM 自报的 0-1 分数")
    print(f"    {'LLM 自报置信度':>16}{'常规 被覆盖':>16}{'例外 被覆盖':>16}")
    for lo,hi in ((0,.7),(.7,.85),(.85,.95),(.95,1.01)):
        m=(CF>=lo)&(CF<hi)
        if m.sum()<10: continue
        a=100*ov[m&~exc].mean() if (m&~exc).sum()>5 else float('nan')
        b=100*ov[m&exc].mean() if (m&exc).sum()>5 else float('nan')
        print(f"    {lo:.2f}–{hi if hi<=1 else 1.0:.2f}      {a:12.1f}% (n={(m&~exc).sum():4d}){b:11.1f}% (n={(m&exc).sum():3d})")
    # 被覆盖的例外, LLM 有多自信?
    print(f"    被覆盖的例外, LLM 平均置信度 = {CF[exc&ov].mean():.3f}  (n={(exc&ov).sum()})")
    print(f"    -> {'⚠ 模型在抹掉例外时是自信的' if CF[exc&ov].mean()>0.8 else '模型在例外上确实不自信'}")
