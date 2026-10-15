"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from . import run, candidate_experiment as ex
from .repeat_context_attention import tokens
from .check_context_clues import parse_name, circuit_key

def clean_input(d):
    ns=[]
    for n in d['nodes']:
        f=n['features']
        g={k:copy.deepcopy(f[k]) for k in ('name','type_code','voltage_kv','winding_voltage_kv')}
        g['bus_name_attributes']=parse_name(g['name'],g['type_code'])
        ns.append({'node_id':n['node_id'],'features':g})
    return {'nodes':ns}

def make_input(d,types):
    d=clean_input(d);b=run.features(d,types,with_labels=False)
    b['tokens']=tokens(d,types);n=len(d['nodes']);fs=[n['features'] for n in d['nodes']]
    keys=[(f['voltage_kv'],circuit_key(f['name'])) for f in fs]
    bias=np.zeros((n,n),np.float32)
    for i in range(n):
        for j in range(n):
            bias[i,j]=.5*float(fs[i]['voltage_kv']==fs[j]['voltage_kv'])
            if keys[i][1] and keys[i]==keys[j] and fs[i]['type_code']!='410' and fs[j]['type_code']!='410':bias[i,j]+=1.5
    b['bias']=bias;b['templates']=ex.group_inputs(d,b)
    return b

def attach_labels(b,original):
    ids={n['node_id']:k for k,n in enumerate(b['d']['nodes'])}
    edges={tuple(sorted((ids[e['source']],ids[e['target']]))) for e in original['labels']['edges']}
    b['y']=np.array([(int(i),int(j)) in edges for i,j in zip(b['i'],b['j'])],np.float32)
    degree=np.zeros(len(ids),np.float32)
    np.add.at(degree,b['i'],b['y']);np.add.at(degree,b['j'],b['y'])
    b['target_degree']=np.log1p(degree)/3
    assert int(b['y'].sum())==len(edges)

def choose_rules(cache,scores,files):
    y=np.concatenate([cache[f]['y'] for f in files]);best=None
    for alpha in (0.,.25,.5,1.,2.):
        z=np.concatenate([np.where(run.structure_mask(cache[f],scores[f]+alpha*cache[f]['name_hint']),scores[f]+alpha*cache[f]['name_hint'],-1e6) for f in files])
        threshold,f1=run.best_threshold(y,z)
        if best is None or f1>best['f1']:best={'alpha':alpha,'threshold':threshold,'f1':f1}
    return best
