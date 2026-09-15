"""分层: 控制住'组合的群体同意率'后, 用户宽容度效应是否仍在?"""
import numpy as np, sys, os
from collections import defaultdict
sys.path.insert(0,'.')
import metrics as M, baselines_ext as B
from data import Graph, build_records, per_user_folds
from train import predict, train
def sp(a,b):
    ra=np.argsort(np.argsort(a)).astype(float); rb=np.argsort(np.argsort(b)).astype(float)
    ra=(ra-ra.mean())/(ra.std()+1e-9); rb=(rb-rb.mean())/(rb.std()+1e-9); return float((ra*rb).mean())
def pp(x,y,r,it=3000):
    rng=np.random.default_rng(0)
    return float((np.array([sp(x,rng.permutation(y)) for _ in range(it)])>=r).mean())
ep=int(os.environ.get("PERM_EPOCHS",200))
conj=lambda r:(r["tool"],r["data_type"])
records,_=build_records(); folds=per_user_folds(records,5,42)

for mlab,fit in (("ranking CF", lambda g,tr,te:(predict(train(g,tr,mode="allow",loss="bpr",use_sideinfo=False,epochs=ep),g,te),
                                                 predict(train(g,tr,mode="allow",loss="bpr",use_sideinfo=False,epochs=ep),g,tr))),
                 ("calibrated FM", lambda g,tr,te:(B.fm_predict(tr,te,objective="bce"), B.fm_predict(tr,tr,objective="bce")))):
    G,V,CR=[],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        g=Graph(tr,records)
        cg=defaultdict(lambda:[0,0])
        for r in tr: a=cg[conj(r)]; a[0]+=r["label"]; a[1]+=1
        pr,prtr=fit(g,tr,te)
        y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
        thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
        yh=(p>=thr).astype(int); u=np.array([r["user_id"] for r in te])
        gr={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
        for i,r in enumerate(te):
            if y[i]!=0: continue
            s,n=cg[conj(r)]
            if n==0: continue
            G.append(gr[r["user_id"]]); V.append(float(yh[i])); CR.append(s/n)
    G,V,CR=np.array(G),np.array(V),np.array(CR)
    print(f"\n### {mlab}  —— 在每个'组合同意率'层内, 大方用户 vs 保守用户")
    print(f"    {'组合同意率层':>16}{'n':>7}{'保守用户':>10}{'大方用户':>10}{'差':>8}{'层内rho':>9}")
    rs=[]
    for lo,hi in ((0,.25),(.25,.5),(.5,.75),(.75,1.01)):
        m=(CR>=lo)&(CR<hi)
        if m.sum()<50: continue
        g_,v_=G[m],V[m]
        loU=100*v_[g_<0.4].mean() if (g_<0.4).any() else float('nan')
        hiU=100*v_[g_>=0.6].mean() if (g_>=0.6).any() else float('nan')
        r=sp(g_,v_); rs.append((r,m.sum()))
        print(f"    {lo:.2f}–{hi if hi<=1 else 1.0:.2f}      {m.sum():7d}{loU:9.1f}%{hiU:9.1f}%{hiU-loU:+7.1f}p{r:+9.3f}")
    if rs:
        wr=sum(r*n for r,n in rs)/sum(n for _,n in rs)
        print(f"    -> 层内加权平均 rho = {wr:+.3f}   "
              f"{'✓ 大方效应独立存在' if wr>0.05 else '✗ 大方效应被组合效应解释掉了'}")
    print(f"    (未分层的 rho = {sp(G,V):+.3f})")
