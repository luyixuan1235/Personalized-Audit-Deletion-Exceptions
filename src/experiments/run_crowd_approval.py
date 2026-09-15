"""大方用户的拒绝是否'反群体'? 纯数据统计, 不训练任何模型."""
import numpy as np, sys, os
sys.path.insert(0,'.')
from data import build_records
from collections import defaultdict

def spearman(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())

records,_=build_records()
tag=os.environ.get("TAG","?")
# 每个请求的"群体同意率"(留一法: 排除该用户自己)
key=lambda r:(r["domain"],r["tool"],r["data_type"])
agg=defaultdict(lambda:[0,0])
for r in records:
    a=agg[key(r)]; a[0]+=r["label"]; a[1]+=1
urate=defaultdict(lambda:[0,0])
for r in records:
    u=urate[r["user_id"]]; u[0]+=r["label"]; u[1]+=1

G,C=[],[]   # 用户grant率, 该拒绝对应的群体同意率(留一)
for r in records:
    if r["label"]!=0: continue
    s,n=agg[key(r)]
    if n<=1: continue
    crowd=s/(n-1)                      # 该用户是 deny(0), 所以留一后分子不变
    gu,nu=urate[r["user_id"]]
    G.append(gu/nu); C.append(crowd)
G,C=np.array(G),np.array(C)
print(f"[{tag}] 共 {len(G)} 个 deny\n")
print(f"  {'用户grant率':>14}{'deny数':>8}{'这些deny对应的群体同意率':>26}")
for lo,hi,lab in ((0,.2,'保守'),(.2,.4,''),(.4,.6,''),(.6,.8,'大方'),(.8,1.01,'很大方')):
    m=(G>=lo)&(G<hi)
    if m.sum()==0: continue
    bar='█'*int(C[m].mean()*40)
    print(f"  {lo:.1f}–{hi if hi<=1 else 1.0:.1f} {lab:6s}{m.sum():8d}      {C[m].mean()*100:5.1f}%  {bar}")
r=spearman(G,C)
rng=np.random.default_rng(0)
null=np.array([spearman(G,rng.permutation(C)) for _ in range(5000)])
p=float((null>=r).mean())
print(f"\n  rho(用户grant率, 其拒绝的群体同意率) = {r:+.3f}   p={p:.4f}")
print(f"  -> {'✓ 大方用户的拒绝确实更"反群体"' if p<0.01 else '✗ 假说不成立'}")
