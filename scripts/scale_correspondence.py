#!/usr/bin/env python3
"""Frozen, paper-local graph/reference scale and correspondence interventions."""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[name]='2'
from pathlib import Path
import json, hashlib, time
import numpy as np
from local_raag import raag_battery
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/scale-correspondence'
METRICS=['reachability_survival','edge_survival','separation_survival','allocation_coherence']

def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,list): return [clean(v) for v in x]
    if isinstance(x,(float,np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x,np.integer): return int(x)
    if isinstance(x,np.bool_): return bool(x)
    return x

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    start=time.time(); meta=json.loads((OUT/'inputs.json').read_text());rows=[];controls=[];corrupt=[];baselines={}
    for item in meta:
        path=OUT/item['file'];assert digest(path)==item['sha256'];a=np.load(path);x=a['reference'];z=a['latent']
        for k in (5,15,30):
            for rank in (10,25,50):
                val=raag_battery(x[:,:rank],z,k=k,already_simplex=item['native'])
                rows.append(dict(id=item['id'],dataset=item['dataset'],seed=item['seed'],native=item['native'],k=k,rank=rank,**val))
                if (k,rank)==(15,50):baselines[item['id']]=val
        # Same orthogonal transform for every row; probabilities no longer native.
        q,_=np.linalg.qr(np.random.default_rng(771).normal(size=(z.shape[1],z.shape[1])))
        val=raag_battery(x,2.5*(np.asarray(z,dtype=float)@q),k=15,already_simplex=False)
        controls.append(dict(id=item['id'],control='rotation_rescale',geometry_delta={m:val[m]-baselines[item['id']][m] for m in METRICS[:3]}))
        ix=np.random.default_rng(772).permutation(len(x));val=raag_battery(x[ix],z[ix],k=15,already_simplex=item['native'])
        keys=METRICS if item['native'] else METRICS[:3]
        controls.append(dict(id=item['id'],control='joint_row_permutation',delta={m:val[m]-baselines[item['id']][m] for m in keys}))
        if item['native']:
            for seed in (0,17,42):
                order=np.random.default_rng(seed).permutation(len(z))
                for fraction in (0,.1,.25,.5,1):
                    n=int(fraction*len(z));idx=np.arange(len(z));chosen=order[:n]
                    if n>=2:idx[chosen]=np.roll(chosen,1)
                    perm=z[idx];assert np.array_equal(np.sort(idx),np.arange(len(z)))
                    val=raag_battery(x,perm,k=15,already_simplex=True)
                    corrupt.append(dict(id=item['id'],dataset=item['dataset'],model_seed=item['seed'],corruption_seed=seed,requested_fraction=fraction,changed_fraction=float(np.mean(idx!=np.arange(len(z)))),**val))
        print(item['id'],'done',flush=True)
    result=clean(dict(scale=rows,correspondence=corrupt,controls=controls,optimizer_updates=0,elapsed_seconds=time.time()-start,code_sha256=digest(Path(__file__)),scorer_sha256=digest(Path(__file__).with_name('local_raag.py')),protocol_sha256=digest(OUT/'PROTOCOL.md')))
    (OUT/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    ranges={}
    for metric in METRICS:
        ranges[metric]=[]
        for item in meta:
            vs=[r[metric] for r in rows if r['id']==item['id'] and np.isfinite(r[metric])]
            if vs:ranges[metric].append(dict(id=item['id'],range=float(max(vs)-min(vs)),minimum=float(min(vs)),maximum=float(max(vs))))
    end={}
    for m in METRICS:
        end[m]={str(f):float(np.nanmean([r[m] for r in corrupt if r['requested_fraction']==f])) for f in (0,.1,.25,.5,1)}
    summary=clean(dict(n_scale=len(rows),n_correspondence=len(corrupt),n_controls=len(controls),scale_ranges=ranges,correspondence_means=end,max_control_abs_delta=max(abs(v) for c in controls for v in c.get('delta',c.get('geometry_delta')).values() if np.isfinite(v)),elapsed_seconds=result['elapsed_seconds']))
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
