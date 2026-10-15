"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from . import run, candidate_experiment as ex
from .whole_station_study import clean_input, attach_labels
from .check_context_clues import circuit_key, parse_name
from .topology_structure_rules import equipment_role

def input_batch(doc,types):
    b=run.features(clean_input(doc),types,with_labels=False)
    ns=b['d']['nodes'];fs=[n['features'] for n in ns]
    ts=[f['type_code'] for f in fs]
    b['bus_mask']=np.array([{ts[i],ts[j]}=={'408','410'} for i,j in zip(b['i'],b['j'])])
    b['bus_groups']={i:np.flatnonzero(b['bus_mask']&((b['i']==i)|(b['j']==i))) for i,t in enumerate(ts) if t=='408'}
    b['hard']={g['ds_index'] for g in ex.group_inputs(b['d'],b)}
    b['lookup']={tuple(sorted((int(i),int(j)))):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    grouped=defaultdict(list)
    for i,f in enumerate(fs):
        key=circuit_key(f['name'])
        if key and f['type_code']!='410':grouped[(f['voltage_kv'],key)].append(i)
    b['circuit_groups']=list(grouped.values())
    b['roles']=[role(f) for f in fs]
    return b

def role(f):
    r=equipment_role(f)
    if r:return r
    s=f['name'];tokens=['主变侧','主变','中性点','正母侧','副母侧','高压','中压','低压','母线侧','线路侧','母联','分段','接地','压变','避雷']
    return f['type_code']+':'+'/'.join(t for t in tokens if t in s)

def decoded_scores(b,z,alpha,decoder):
    s=z+alpha*b['name_hint']
    if decoder!='raw':s=np.where(run.structure_mask(b,s),s,-1e6)
    if decoder=='exclusive':
        s=s.copy()
        for idx in b['bus_groups'].values():
            if not len(idx):continue
            winners=idx[s[idx]==s[idx].max()];maximum=float(s[idx].max());s[idx]=-1e6
            if len(winners)==1:s[winners[0]]=maximum
    return s

def select(cache,files,scores,decoder):
    yy=np.concatenate([cache[f]['y'] for f in files]);best=None
    for alpha in ([0.] if decoder=='raw' else [0.,.25,.5,1.,2.]):
        z=np.concatenate([decoded_scores(cache[f],scores[f],alpha,decoder) for f in files])
        threshold,vf=run.best_threshold(yy,z)
        if best is None or vf>best['validation_f1']:best={'alpha':alpha,'threshold':threshold,'validation_f1':vf}
    return best

def sample_training(cache,files,seed):
    rng=np.random.default_rng(seed);xx=[];yy=[]
    for fn in files:
        b=cache[fn];pos=np.flatnonzero(b['y']);neg=np.flatnonzero(1-b['y'])
        idx=np.r_[pos,rng.choice(neg,min(len(neg),5*len(pos)),replace=False)]
        xx.append(b['x'][idx]);yy.append(b['y'][idx])
    return np.concatenate(xx),np.concatenate(yy)
