"""SPA: the side we never measured.

Every SPA measurement in this project examined REFUSALS. But SPA is deny-majority
(grant rate 0.36), and the two-prior mechanism predicts the double-contradiction
cell -- the harmed corner -- lives on the CONSENT side there: a restrictive user's
rare grant of something the crowd also denies. If the mechanism is right, SPA's
2x2 should be monotone like Wu's with the hot corner flipped to consents; if SPA
is just 'weak', both sides should be flat.
"""
import numpy as np, sys, os
from collections import defaultdict
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/experiments')
import _common as C
import metrics as M
from data import Graph, build_records, per_user_folds
from train import predict, train

ep=15 if C.QUICK else int(os.environ.get("PERM_EPOCHS",200))
records,_=build_records(); folds=per_user_folds(records,5,42)
conj=lambda r:(r["tool"],r["data_type"])

UM,CM,Y,YH=[],[],[],[]
for f in range(5):
    tr=[records[i] for i in range(len(records)) if folds[i]!=f]
    te=[records[i] for i in range(len(records)) if folds[i]==f]
    g=Graph(tr,records)
    m=train(g,tr,mode="allow",loss="bpr",use_sideinfo=False,epochs=ep)
    pr,prtr=predict(m,g,te),predict(m,g,tr)
    y=np.asarray(pr["y"]); p=np.asarray(pr["p_allow"])
    thr=float(M.equal_error_threshold(np.asarray(prtr["y"]),np.asarray(prtr["p_allow"])))
    yh=(p>=thr).astype(int)
    u=np.array([r["user_id"] for r in te])
    ur={uu: float(y[u==uu].mean()) for uu in np.unique(u)}
    mu=np.array([1 if ur[uu]>=0.5 else 0 for uu in u])
    cg=defaultdict(lambda:[0,0])
    for r in tr: a=cg[conj(r)]; a[0]+=r["label"]; a[1]+=1
    cm=np.array([1 if (cg[conj(r)][0]/max(cg[conj(r)][1],1))>=0.5 else 0 for r in te])
    Y+=list(y); YH+=list(yh); UM+=list(mu); CM+=list(cm)
Y,YH,UM,CM=map(np.array,(Y,YH,UM,CM))
ov=(YH!=Y)

tag=os.environ.get("TAG","?")
print(f"[{tag}]  n={len(Y)}  corpus grant rate={Y.mean():.2f}\n")
print("== EOR (both directions) ==")
exc=(Y!=UM)
print(f"  routine {100*ov[~exc].mean():5.1f}%   exception {100*ov[exc].mean():5.1f}%   ratio {ov[exc].mean()/max(ov[~exc].mean(),1e-9):.1f}x")
print("\n== symmetric split ==")
pr_=exc&(Y==0); rc=exc&(Y==1)
print(f"  permissive users' rare REFUSALS erased: {100*ov[pr_].mean():5.1f}%  (n={pr_.sum()})")
print(f"  restrictive users' rare CONSENTS erased: {100*ov[rc].mean():5.1f}%  (n={rc.sum()})")
print("\n== the 2x2 (own majority x crowd) ==")
print(f"  {'':24}{'agrees crowd':>14}{'contradicts crowd':>19}")
_OUT={}
for au,lab in ((True,'agrees own habit'),(False,'CONTRADICTS own habit')):
    row=f"  {lab:24}"
    for ac in (True,False):
        m_=((Y==UM)==au)&((Y==CM)==ac)
        row+=f"{100*ov[m_].mean():10.1f}% (n={m_.sum():5d})" if m_.sum()>20 else f"{'-':>14}"
        if m_.sum()>20:
            _OUT[('agree_habit' if au else 'contra_habit')+'/'+('agree_crowd' if ac else 'contra_crowd')]=dict(override=float(100*ov[m_].mean()),n=int(m_.sum()))
    print(row)
print("\n预言: deny-majority 语料的受害角 = 违背习惯的 CONSENT x 违背群体 (右下角)")

import json as _json, os as _os
_json.dump(_OUT, open(_os.path.join(_os.environ.get("PERM_RESULTS_DIR","../results_spa"),"table54_spa_symmetric.json"),"w"), indent=1)
print("saved table54_spa_symmetric.json")
