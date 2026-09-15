"""在 Wu 真正部署的三个模型上验证双机制."""
import json, numpy as np, sys
from collections import defaultdict
sys.path.insert(0,'.')
from analyze_wu_peruser import load_cf, load_llm

def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())

SRC=[('CF only (LightGCN+BPR)', lambda: load_cf('../results/cf_only_predictions.json')[:3]),
     ('IC only (LLM)',          lambda: load_llm('../results/ic_only_predictions.json')[:3]),
     ('IC+CF (部署系统, 85.1%)', lambda: load_llm('../results/ic_cf_predictions.json')[:3])]

# 用完整数据集算每个请求(itemID/data_type)的群体同意率
full=json.load(open('../data/processed_dataset.json'))

for name,f in SRC:
    users,y,yhat=f()
    # 该请求类型在其他用户中的同意率(留一)
    agg=defaultdict(lambda:[0,0])
    # 这里用 itemID / (tool,data) 作为组合键: 从 users/y 重建
    # load_cf 返回 users,y,yhat; 需要 item 键 -> 从原始文件重读
    if 'CF' in name and 'IC' not in name:
        d=json.load(open('../results/cf_only_predictions.json'))['cf_only']['test_predictions']
        keys=[r['itemID'] for r in d]
    else:
        fn='../results/ic_only_predictions.json' if 'IC only' in name else '../results/ic_cf_predictions.json'
        d=json.load(open(fn)); keys=[]
        for pid,dat in d.items():
            if not isinstance(dat,dict) or dat.get('skipped') or 'error' in dat: continue
            preds={p.get('id'):p.get('permission',{}) for p in dat.get('predictions',[])}
            for gt in dat.get('ground_truth',[]):
                pp=preds.get(gt['id'])
                if not pp: continue
                for dt in gt.get('answer',{}):
                    if dt in pp: keys.append(dt)
    keys=np.array(keys[:len(y)])
    if len(keys)!=len(y): continue
    for k,yy in zip(keys,y):
        a=agg[k]; a[0]+=int(yy); a[1]+=1
    CR=np.array([ (agg[k][0]-yy)/max(agg[k][1]-1,1) for k,yy in zip(keys,y) ])
    gr={u: float(y[users==u].mean()) for u in np.unique(users)}
    G=np.array([gr[u] for u in users])
    dn=(y==0)
    G,V,CR=G[dn],yhat[dn].astype(float),CR[dn]
    print(f"\n### {name}   ({len(G)} denials)")
    print(f"    {'群体同意率层':>14}{'n':>6}{'保守':>9}{'大方':>9}{'差':>8}")
    rs=[]
    for lo,hi in ((0,.33),(.33,.66),(.66,1.01)):
        m=(CR>=lo)&(CR<hi)
        if m.sum()<20: continue
        g_,v_=G[m],V[m]
        a=100*v_[g_<0.4].mean() if (g_<0.4).any() else float('nan')
        b=100*v_[g_>=0.6].mean() if (g_>=0.6).any() else float('nan')
        rs.append((sp(g_,v_),m.sum()))
        print(f"    {lo:.2f}–{hi if hi<=1 else 1.0:.2f}   {m.sum():6d}{a:8.1f}%{b:8.1f}%{b-a:+7.1f}p")
    print(f"    rho(群体同意率, 被推翻) = {sp(CR,V):+.3f}")
    if rs:
        wr=sum(r*n for r,n in rs)/sum(n for _,n in rs)
        print(f"    层内加权 rho(用户宽容度) = {wr:+.3f}   (未分层 {sp(G,V):+.3f})")
