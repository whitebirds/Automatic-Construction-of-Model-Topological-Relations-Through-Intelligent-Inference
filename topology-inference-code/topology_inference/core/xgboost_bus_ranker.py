"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from . import external_baselines as base
from .bus_exclusive_study import groups_from_inputs

def inputs(doc,types):
    b=base.input_batch(doc,types);groups=groups_from_inputs(b,types)
    for g in groups:
        ix=g['indices'];g['x']=np.c_[np.vstack([b['x'][ix],np.zeros((1,b['x'].shape[1]),np.float32)]),g['context']]
        g['hard']=g['device'] in b['hard']
    b['groups']=groups
    return b

def attach(b,doc):
    base.attach_labels(b,doc)
    for g in b['groups']:
        y=b['y'][g['indices']];assert y.sum()<=1
        g['y']=np.r_[y,1-y.sum()].astype(np.float32)

def matrix(cache,files,labels):
    import xgboost as xgb
    gs=[g for fn in files for g in cache[fn]['groups']];x=np.concatenate([g['x'] for g in gs]);dm=xgb.DMatrix(x,label=np.concatenate([g['y'] for g in gs]) if labels else None,nthread=2)
    dm.set_group([len(g['x']) for g in gs]);return dm

def choices(cache,files,z):
    result={};offset=0;hard_errors=errors=0
    for fn in files:
        b=cache[fn];pred=np.zeros(len(b['y']),bool)
        for g in b['groups']:
            zz=z[offset:offset+len(g['x'])];offset+=len(zz);w=np.flatnonzero(np.isclose(zz,zz.max(),rtol=0,atol=1e-9));pick=int(w[0]) if len(w)==1 else len(zz)-1
            if pick<len(g['indices']):pred[g['indices'][pick]]=True
            wrong=not bool(g['y'][pick]);errors+=wrong;hard_errors+=int(wrong and g['hard'])
        result[fn]=pred
    assert offset==len(z)
    return result,(hard_errors,errors)
