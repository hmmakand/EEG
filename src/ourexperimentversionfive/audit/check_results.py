"""Read-only audit of saved runs and fixed, untuned linear feature baselines.
Run from repository root: python src/ourexperimentversionfive/audit/check_results.py
"""
import json, hashlib
from pathlib import Path
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
ROOT = Path(__file__).resolve().parents[1]
D = Path('data/moabb/Graph-Liu2024-VersionTwo')
out = {'runs': {}}
for p in sorted((ROOT/'outputs').iterdir()):
    if not (p/'within_subject_summary.json').exists(): continue
    folds = [f for q in sorted(p.glob('subject_*/within_subject_results.json')) for f in json.loads(q.read_text())['folds']]
    a=np.array([f['history'][-1]['training']['accuracy'] for f in folds]); e=np.array([f['evaluation']['accuracy'] for f in folds]); ep=np.array([f['best_epoch'] for f in folds])
    for f in folds:
        tr=set(f['preprocessing']['train_graph_indices']); te=set(f['preprocessing']['evaluation_graph_indices']); assert not tr & te
    s=json.loads((p/'within_subject_summary.json').read_text())
    out['runs'][p.name]={'summary':{k:v for k,v in s.items() if k!='per_subject'}, 'last_online_train_accuracy':a.mean(), 'last_online_train_loss':np.mean([f['history'][-1]['training']['loss'] for f in folds]), 'online_train_eval_gap':(a-e).mean(), 'train_at_least_90_count':int((a>=.9).sum()), 'train_at_most_60_count':int((a<=.6).sum()), 'selected_epoch_quantiles':np.quantile(ep,[0,.25,.5,.75,1]).tolist(), 'selected_epoch_le2':int((ep<=2).sum()), 'single_class_prediction_folds':sum(f['evaluation']['accuracy']==.5 and f['evaluation']['recall'] in (0,1) for f in folds), 'high_train_subset_eval_accuracy':float(e[a>=.9].mean()) if (a>=.9).any() else None}
x=np.load(D/'node_features_without_csd.npy'); y=np.load(D/'labels.npy'); edge=np.load(D/'edge_attr_wpli_without_csd.npy')[:,:,2]; ei=np.load(D/'edge_index.npy')
import csv
with (D/'samples.tsv').open() as f: samples=list(csv.DictReader(f,delimiter='\t'))
subjects=np.array([int(s['subject']) for s in samples])
out['data']={'shape':list(x.shape),'labels':np.bincount(y).tolist(),'finite':bool(np.isfinite(x).all() and np.isfinite(edge).all()),'feature_mean':x.mean((0,1)).tolist(),'feature_std':x.std((0,1)).tolist(),'feature_min':x.min((0,1)).tolist(),'feature_max':x.max((0,1)).tolist(),'duplicate_node_trials':len(x)-len({hashlib.sha256(t.tobytes()).hexdigest() for t in x}), 'edge_quantiles':np.quantile(edge,[0,.25,.5,.75,1]).tolist()}
A=np.zeros((len(x),29,29)); A[:,ei[0],ei[1]]=edge; A+=np.eye(29); deg=A.sum(-1); P=A/np.sqrt(deg[:,:,None]*deg[:,None,:]); z=(x-x.mean((0,1)))/x.std((0,1)); mixed=P@z
out['data']['spatial_variance_ratio_after_normalized_adjacency']= (mixed.var(1).mean(0)/z.var(1).mean(0)).tolist()
out['baselines']={}
for name,xx in [('node_flat',x.reshape(len(x),-1)),('alpha_beta_flat',x[:,:,2:4].reshape(len(x),-1)),('alpha_edges',edge[:,:406])]:
    acc=[]; auc=[]
    for sub in np.unique(subjects):
        idx=np.flatnonzero(subjects==sub); pred=np.zeros(40); scores=np.zeros(40)
        for tr,te in StratifiedKFold(5,shuffle=True,random_state=42).split(idx,y[idx]):
            model=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000));model.fit(xx[idx[tr]],y[idx[tr]]);pred[te]=model.predict(xx[idx[te]]);scores[te]=model.predict_proba(xx[idx[te]])[:,1]
            auc.append(roc_auc_score(y[idx[te]],scores[te]))
        acc.append(accuracy_score(y[idx],pred))
    out['baselines'][name]={'mean_accuracy':np.mean(acc),'mean_fold_auc':np.mean(auc),'per_subject_accuracy':acc}
(ROOT/'audit/evidence.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({**out,'baselines':{k:{a:b for a,b in v.items() if a!='per_subject_accuracy'} for k,v in out['baselines'].items()}},indent=2))
