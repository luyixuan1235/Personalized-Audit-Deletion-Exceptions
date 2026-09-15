"""按 Wu 表的粒度 (data type, 跨 tool 聚合) 重测."""
import json, numpy as np
from collections import defaultdict
def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())
d=json.load(open('../results/ic_cf_predictions.json'))
Y,YH,K=[],[],[]
for pid,dat in d.items():
    if not isinstance(dat,dict) or dat.get('skipped') or 'error' in dat: continue
    preds={p.get('id'):p.get('permission',{}) for p in dat.get('predictions',[])}
    for gt in dat.get('ground_truth',[]):
        pp=preds.get(gt['id'])
        if not pp: continue
        for dt,gl in gt.get('answer',{}).items():
            if dt not in pp: continue
            k=dt.split(',',1)[-1].strip()          # 数据类型部分, 跨 tool 聚合 (他们表的粒度)
            Y.append(1 if 'Yes' in gl else 0); YH.append(1 if 'Yes' in pp[dt].get('label','') else 0); K.append(k)
Y,YH,K=np.array(Y),np.array(YH),np.array(K)
rows=[]
for t in np.unique(K):
    m=(K==t)
    if m.sum()<10: continue
    g=Y[m].mean(); gr=m&(Y==1); dn=m&(Y==0)
    fnr=1-YH[gr].mean() if gr.sum()>=4 else np.nan
    fpr=YH[dn].mean() if dn.sum()>=4 else np.nan
    rows.append((t,g,fnr,fpr,int(m.sum())))
G=np.array([r[1] for r in rows]); FNR=np.array([r[2] for r in rows]); FPR=np.array([r[3] for r in rows])
mf=~np.isnan(FNR); mp=~np.isnan(FPR)
print(f"可测类型: FNR 侧 {mf.sum()} 个, FPR 侧 {mp.sum()} 个 (共 {len(rows)} 类)\n")
rng=np.random.default_rng(0)
r1=sp(G[mf],FNR[mf]); p1=float((np.array([sp(G[mf],rng.permutation(FNR[mf])) for _ in range(5000)])<=r1).mean())
r2=sp(G[mp],FPR[mp]); p2=float((np.array([sp(G[mp],rng.permutation(FPR[mp])) for _ in range(5000)])>=r2).mean())
print(f"rho(类型群体同意率, FNR) = {r1:+.3f}  p={p1:.4f}   [预言: 负]")
print(f"rho(类型群体同意率, FPR) = {r2:+.3f}  p={p2:.4f}   [预言: 正]")

# 稳健性: 并列秩校正 (scipy) x 最小格子数 —— 见附录 "Footprints, in full"
try:
    from scipy.stats import spearmanr
    print("\n稳健性 (tie-corrected Spearman, 最小格子数扫描):")
    for minn, minc in [(5,2),(8,3),(10,4)]:
        G2,FNR2,FPR2=[],[],[]
        for t in np.unique(K):
            m=(K==t)
            if m.sum()<minn: continue
            gr=m&(Y==1); dn=m&(Y==0)
            G2.append(Y[m].mean())
            FNR2.append(1-YH[gr].mean() if gr.sum()>=minc else np.nan)
            FPR2.append(YH[dn].mean() if dn.sum()>=minc else np.nan)
        G2,FNR2,FPR2=np.array(G2),np.array(FNR2),np.array(FPR2)
        a,b=~np.isnan(FNR2),~np.isnan(FPR2)
        s1,q1=spearmanr(G2[a],FNR2[a]); s2,q2=spearmanr(G2[b],FPR2[b])
        print(f"  min n={minn:2d}: FNR {s1:+.3f} (p={q1:.4f}, n={a.sum()})  FPR {s2:+.3f} (p={q2:.4f}, n={b.sum()})")
except ImportError:
    pass
# 展示最极端的几类
rows.sort(key=lambda r:r[1])
print(f"\n{'data type':40s}{'同意率':>7}{'FNR':>7}{'FPR':>7}{'n':>5}")
for t,g,fnr,fpr,n in rows[:5]+rows[-5:]:
    f1='  -' if np.isnan(fnr) else f'{100*fnr:4.0f}%'; f2='  -' if np.isnan(fpr) else f'{100*fpr:4.0f}%'
    print(f"{t[:40]:40s}{g:7.2f}{f1:>7}{f2:>7}{n:>5}")
