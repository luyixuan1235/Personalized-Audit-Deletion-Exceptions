"""Three-state routing (subjective-logic opinion triangle), the last untested form.

b/d/u is already the three-state formulation; vacuity-u and conflict were each tested
alone and lost to the margin. The remaining question: does a 2-D deferral REGION on the
opinion triangle (margin-low OR conflict-high OR vacuity-high), at the SAME 40% budget,
beat the 1-D margin? Also diagnostic: among decisions the margin KEEPS, does high
conflict still mark elevated exception override (complementarity), or is conflict's
information already inside the margin?
"""
import numpy as np, sys, os
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/experiments')
import _common as C, metrics as M
from data import Graph, build_records, per_user_folds
from train import predict, train

ep=15 if C.QUICK else int(os.environ.get("PERM_EPOCHS",200))
records,_=build_records(); folds=per_user_folds(records,5,42)
Y,YH,EX,MG,CN,VU=[],[],[],[],[],[]
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
    MG+=list(np.abs(p-thr)); CN+=list(2*np.minimum(ea,ed)/(ea+ed+1e-9)); VU+=list(np.asarray(pr["u"]))
Y,YH,EX,MG,CN,VU=map(np.array,(Y,YH,EX,MG,CN,VU)); ov=(YH!=Y); n=len(Y)

def rank(x):            # 秩归一化, 高=更该弃权
    return np.argsort(np.argsort(x))/(len(x)-1)
r_mg=rank(-MG); r_cn=rank(CN); r_vu=rank(VU)

def eval_defer(defer,lab):
    auto=~defer
    eo=100*ov[auto&(EX==1)].mean(); dn=auto&(Y==0); fgr=100*YH[dn].mean()
    cap=100*EX[defer].sum()/EX.sum()
    print(f"{lab:34s}{100*auto.mean():7.0f}%{cap:9.1f}%{eo:10.1f}%{fgr:9.1f}%")

BUD=0.40
print(f"{'deferral rule (40% budget)':34s}{'cover':>8}{'例外捕获':>10}{'自动集例外':>11}{'FGR':>8}")
eval_defer(r_mg>=1-BUD, "margin only")
for a in (0.30,0.20,0.10):
    b=BUD-a
    d = (r_mg>=1-a)|(r_cn>=1-b)
    # 并集会少于预算, 用margin补满
    need=int(BUD*n)-d.sum()
    if need>0:
        idx=np.argsort(-r_mg); extra=[i for i in idx if not d[i]][:need]; d[extra]=True
    eval_defer(d, f"margin {int(a*100)}% ∪ conflict {int(b*100)}%")
d=(r_mg>=1-0.25)|(r_cn>=1-0.10)|(r_vu>=1-0.05)
need=int(BUD*n)-d.sum()
if need>0:
    idx=np.argsort(-r_mg); extra=[i for i in idx if not d[i]][:need]; d[extra]=True
eval_defer(d, "triangle: mg25 ∪ cn10 ∪ vu5")

print("\n== 互补性诊断: margin 保留的决策中, 高冲突子集是否仍富含被抹例外? ==")
keep=r_mg<1-BUD
hi=keep&(r_cn>=0.8); lo=keep&(r_cn<0.8)
print(f"  保留集高冲突 (top20%): 例外覆盖 {100*ov[hi&(EX==1)].mean():5.1f}%  (n_exc={int((hi&(EX==1)).sum())})")
print(f"  保留集低冲突         : 例外覆盖 {100*ov[lo&(EX==1)].mean():5.1f}%  (n_exc={int((lo&(EX==1)).sum())})")
