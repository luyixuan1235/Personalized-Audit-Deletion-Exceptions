"""The user's original idea, tested in its correct form.

EDL's u = 2/(e_a+e_d+2) measures TOTAL evidence, so it calls a CONFLICT (both channels
firing) "confident" -- which is exactly the epistemic state an exception occupies: the
habit pumps e_allow while the request pumps e_deny. That may be why u failed as a
deferral signal, without the two-channel idea itself being at fault.

The idea's correct form: defer on DISSONANCE -- both separately-modelled channels
firing at once. Same model, same predictions, four deferral signals:
  margin      |p - thr|                 (defer low)
  u           total-evidence vacuity    (defer high)   <- failed before
  dissonance  min(e_a, e_d)             (defer high)   <- the idea, done right
  balance     2*min/(e_a+e_d)           (defer high)
"""
import numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/experiments')
import _common as C, metrics as M
from data import Graph, build_records, per_user_folds
from train import predict, train

ep=15 if C.QUICK else int(os.environ.get("PERM_EPOCHS",200))
records,_=build_records(); folds=per_user_folds(records,5,42)
Y,YH,EX,MG,U_,DS,BL=[],[],[],[],[],[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    g=Graph(tr,records)
    m=train(g,tr,mode="dual",loss="evi_refuse",use_sideinfo=True,epochs=ep)
    pr,prtr=predict(m,g,te),predict(m,g,tr)
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    ea,ed=np.asarray(pr["e_allow"]),np.asarray(pr["e_deny"])
    thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
    yh=(p>=thr).astype(int)
    u=np.array([r["user_id"] for r in te])
    ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
    mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u])
    Y+=list(y); YH+=list(yh); EX+=list((y!=mu).astype(int))
    MG+=list(np.abs(p-thr))                       # 高=自信
    U_+=list(-np.asarray(pr["u"]))                # 高=自信 (u 低)
    DS+=list(-np.minimum(ea,ed))                  # 高=自信 (冲突低)
    BL+=list(-(2*np.minimum(ea,ed)/(ea+ed+1e-9))) # 高=自信 (平衡度低)
Y,YH,EX=map(np.array,(Y,YH,EX)); ov=(YH!=Y)
print(f"{'deferral signal':18s}{'coverage':>9}{'例外捕获':>10}{'自动集例外':>11}{'自动集FGR':>10}")
for sig,lab in ((np.array(MG),'margin'),(np.array(U_),'u (vacuity)'),
                (np.array(DS),'dissonance min'),(np.array(BL),'balance')):
    for cov in (1.0,0.60):
        auto=np.ones(len(sig),bool) if cov>=1.0 else sig>=np.quantile(sig,1-cov)
        cap=100*EX[~auto].sum()/max(EX.sum(),1)
        eo=100*ov[auto&(EX==1)].mean() if (auto&(EX==1)).sum()>10 else float('nan')
        dn=auto&(Y==0); fgr=100*YH[dn].mean() if dn.sum()>10 else float('nan')
        print(f"{lab:18s}{100*cov:8.0f}%{cap:9.1f}%{eo:10.1f}%{fgr:9.1f}%")
