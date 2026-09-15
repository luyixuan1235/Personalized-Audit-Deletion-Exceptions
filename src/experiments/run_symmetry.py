"""检验1 对称性: 保守用户罕见的 GRANT 是否也被系统性误判为 DENY?"""
import numpy as np, sys
sys.path.insert(0,'.')
from analyze_wu_peruser import load_cf, load_llm
def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())
def pv(x,y,r,it=3000):
    rng=np.random.default_rng(0)
    return float((np.array([sp(x,rng.permutation(y)) for _ in range(it)])>=r).mean())
SRC=[('CF only (LightGCN+BPR)', lambda: load_cf('../results/cf_only_predictions.json')[:3]),
     ('IC only (LLM)',          lambda: load_llm('../results/ic_only_predictions.json')[:3]),
     ('IC+CF (deployed)',       lambda: load_llm('../results/ic_cf_predictions.json')[:3])]
for name,f in SRC:
    u,y,yh=f()
    gr={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
    G=np.array([gr[uu] for uu in u])
    dn=(y==0); Gd,Vd=G[dn],yh[dn].astype(float)          # deny 被推翻为 grant
    gt=(y==1); Gg,Vg=G[gt],(1-yh[gt]).astype(float)      # grant 被误判为 deny
    print(f"\n### {name}")
    print(f"    {'用户grant率':>13}{'deny→被授权':>18}{'grant→被拒绝':>18}")
    for lo,hi,lab in ((0,.2,'保守'),(.2,.4,''),(.4,.6,''),(.6,.8,''),(.8,1.01,'大方')):
        md=(Gd>=lo)&(Gd<hi); mg=(Gg>=lo)&(Gg<hi)
        a=f"{100*Vd[md].mean():5.1f}% (n={md.sum():3d})" if md.sum()>5 else "     -      "
        b=f"{100*Vg[mg].mean():5.1f}% (n={mg.sum():3d})" if mg.sum()>5 else "     -      "
        print(f"    {lo:.1f}–{hi if hi<=1 else 1.0:.1f} {lab:4s} {a:>17}{b:>18}")
    rd=sp(Gd,Vd); pd_=pv(Gd,Vd,rd)
    rg=sp(-Gg,Vg); pg=pv(-Gg,Vg,rg)
    print(f"    rho(越大方, deny被推翻)   = {rd:+.3f}  p={pd_:.4f}   [已知效应]")
    print(f"    rho(越保守, grant被误拒)  = {rg:+.3f}  p={pg:.4f}   [对称检验]")
    sym = rg>0.05 and pg<0.01
    print(f"    -> {'✓ 对称 -> per-user majority reinforcement' if sym else '✗ 不对称 -> 只有 grant 方向被过度预测'}")
