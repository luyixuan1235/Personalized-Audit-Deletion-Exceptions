"""Cross-model check of the IC component (reviewer request): o4-mini vs gpt-4o-mini.
Reads Wu-repo prediction JSONs; writes results/table32_crossmodel_ic.json.
Run permission_ic_only.py with OPENAI_MODEL=gpt-4o-mini first (predictions saved as
ic_only_gpt4omini_predictions.json)."""
import json, sys, numpy as np

def load(path):
    d = json.load(open(path)); users,y,yh,cf=[],[],[],[]
    for pid, blob in d.items():
        for pred, gt in zip(blob["predictions"], blob["ground_truth"]):
            pp = pred.get("permission", {})
            for dtype, gt_label in gt.get("answer", {}).items():
                if dtype not in pp: continue
                users.append(pid); y.append(1 if "Yes" in gt_label else 0)
                yh.append(1 if "Yes" in pp[dtype].get("label","") else 0)
                cf.append(float(pp[dtype].get("score", np.nan)))
    return map(np.array,(users,y,yh,cf))

def report(path,name):
    u,y,yh,cf=load(path)
    gr={x:y[u==x].mean() for x in np.unique(u)}
    mu=np.array([1 if gr[x]>=0.5 else 0 for x in u]); exc=y!=mu; ov=yh!=y
    defer=[]
    for s in range(20):
        rng=np.random.default_rng(s)
        order=np.lexsort((rng.random(len(cf)), cf))
        keep=np.zeros(len(cf),bool); keep[order[int(0.4*len(cf)):]]=True
        defer.append(ov[keep&exc].mean())
    return dict(n=int(len(y)), n_exc=int(exc.sum()), acc=float((yh==y).mean()),
        eor_exc=float(ov[exc].mean()), eor_rout=float(ov[~exc].mean()),
        conf_erasing=float(cf[exc&ov].mean()), conf_overall=float(cf.mean()),
        defer60_exc_tierobust=float(np.mean(defer)))

if __name__=='__main__':
    base=sys.argv[1] if len(sys.argv)>1 else '../wu-repo/results'
    out={n: report(f'{base}/{f}',n) for n,f in
         [('o4-mini','ic_only_predictions.json'),
          ('gpt-4o-mini','ic_only_gpt4omini_predictions.json'),
          ('hybrid_o4mini','ic_cf_predictions.json')]}
    json.dump(out, open('../results/table32_crossmodel_ic.json','w'), indent=1)
    print(json.dumps(out, indent=1))
