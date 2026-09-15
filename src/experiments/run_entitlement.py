"""Does the original signed/evidential design earn a place as an ENTITLEMENT DETECTOR?

The paper's surviving mitigation is deferral, and deferral is only as good as the
confidence it defers on. A single-score model conflates "evidence of grant" with
"no evidence of refusal"; the dual-channel design was built to separate them, and its
learned uncertainty u = 2/(e_a+e_d+2) is an explicit "am I entitled to predict?" signal.

Decisive test: at equal automation coverage, which confidence source (i) defers the most
EXCEPTIONS and (ii) leaves the lowest exception override rate on the automated set?
  fm-margin      : calibrated FM, |p - thr|
  signed-margin  : calibrated signed model, |p - thr|
  edl-u          : dual-graph evidential, defer highest learned uncertainty u
If edl-u/signed do not beat fm-margin on exceptions, the original idea is fully dead
and we say so.
"""
import numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/experiments')
import _common as C, metrics as M, baselines_ext as B
from data import Graph, build_records, per_user_folds
from train import predict, train

ep=15 if C.QUICK else int(os.environ.get("PERM_EPOCHS",200))
records,_=build_records(); folds=per_user_folds(records,5,42)

def collect(kind):
    Y,YH,CF,EX=[],[],[],[]
    for f in range(5):
        tr=[records[i] for i in range(len(records)) if folds[i]!=f]
        te=[records[i] for i in range(len(records)) if folds[i]==f]
        g=Graph(tr,records)
        if kind=='fm':
            pr,prtr=B.fm_predict(tr,te,objective="bce"),B.fm_predict(tr,tr,objective="bce"); u_unc=None
        elif kind=='signed':
            m=B.train_signed(g,tr,epochs=ep,use_sideinfo=True,objective="bce")
            pr,prtr=predict(m,g,te),predict(m,g,tr); u_unc=None
        else:
            m=train(g,tr,mode="dual",loss="evi_refuse",use_sideinfo=True,epochs=ep)
            pr,prtr=predict(m,g,te),predict(m,g,tr); u_unc=np.asarray(pr["u"])
        y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
        thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
        yh=(p>=thr).astype(int)
        conf = -u_unc if u_unc is not None else np.abs(p-thr)   # 高=自信
        u=np.array([r["user_id"] for r in te])
        ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
        mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u])
        Y+=list(y); YH+=list(yh); CF+=list(conf); EX+=list((y!=mu).astype(int))
    return map(np.array,(Y,YH,CF,EX))

print(f"{'confidence source':22s}{'coverage':>9}{'例外弃权捕获率':>13}{'自动集例外覆盖':>14}{'自动集FGR':>11}")
for kind,lab in (('fm','fm-margin'),('signed','signed-margin'),('edl','edl learned-u')):
    Y,YH,CF,EX=collect(kind)
    ov=(YH!=Y)
    for cov in (1.0,0.75,0.60):
        auto=np.ones(len(CF),bool) if cov>=1.0 else CF>=np.quantile(CF,1-cov)
        cap=100*EX[~auto].sum()/max(EX.sum(),1)                 # 例外被弃权的比例
        eo=100*ov[auto&(EX==1)].mean() if (auto&(EX==1)).sum()>10 else float('nan')
        dn=auto&(Y==0); fgr=100*YH[dn].mean() if dn.sum()>10 else float('nan')
        print(f"{lab:22s}{100*cov:8.0f}%{cap:12.1f}%{eo:13.1f}%{fgr:10.1f}%")
