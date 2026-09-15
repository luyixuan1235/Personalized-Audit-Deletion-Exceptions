"""例外在留出设置下是否'可预测'? 即: 训练集里有没有能推出这次例外的信号?"""
import numpy as np, sys
from collections import defaultdict
sys.path.insert(0,'.')
from data import build_records, per_user_folds
records,_=build_records(); folds=per_user_folds(records,5,42)

tot=defaultdict(int); hit=defaultdict(int)
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    # 用户在训练集里的多数行为
    ur=defaultdict(lambda:[0,0])
    for r in tr:
        a=ur[r["user_id"]]; a[0]+=r["label"]; a[1]+=1
    mu={u:(1 if a[0]/max(a[1],1)>=0.5 else 0) for u,a in ur.items()}
    # 该用户在训练集里的"例外"(违背自己多数的决策), 按属性索引
    exc_dt=defaultdict(set); exc_dom=defaultdict(set); exc_tool=defaultdict(set)
    for r in tr:
        u=r["user_id"]
        if u in mu and r["label"]!=mu[u]:
            exc_dt[(u,r["label"])].add(r["data_type"])
            exc_dom[(u,r["label"])].add(r["domain"])
            exc_tool[(u,r["label"])].add(r["tool"])
    for r in te:
        u=r["user_id"]
        if u not in mu or r["label"]==mu[u]: continue      # 只看测试集里的例外
        k=(u,r["label"])
        tot['all']+=1
        if r["data_type"] in exc_dt[k]: hit['same data_type']+=1
        if r["domain"]   in exc_dom[k]: hit['same domain']+=1
        if r["tool"]     in exc_tool[k]: hit['same tool']+=1
        if (r["data_type"] in exc_dt[k]) or (r["domain"] in exc_dom[k]) or (r["tool"] in exc_tool[k]):
            hit['any of the three']+=1
        if len(exc_dt[k])>0: hit['has ANY prior exception']+=1

n=tot['all']
print(f"测试集中的例外共 {n} 个\n")
print("在训练集里, 该用户是否有'同方向的例外'与之共享属性?")
print(f"  {'该用户在训练集里有过任何同向例外':38s}{hit['has ANY prior exception']:5d}  ({100*hit['has ANY prior exception']/n:5.1f}%)")
for k in ('same data_type','same domain','same tool','any of the three'):
    print(f"  {k:38s}{hit[k]:5d}  ({100*hit[k]/n:5.1f}%)")
print(f"\n=> {100*(1-hit['any of the three']/n):.1f}% 的例外, 在训练集里找不到任何同属性的先例")
print("   这些例外在留出设置下是信息论上不可预测的 —— 没有模型能救.")
