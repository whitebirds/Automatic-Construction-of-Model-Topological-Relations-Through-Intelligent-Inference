"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .dedicated_bus_study import inputs, targets, assign
from .dedicated_bus_network import softmax

def prepare(doc,enc):
    p,v,models=enc;d=inputs(doc,p['types'],v);zs=[];hs=[];es=[]
    for m,s in models:
        d['node_x']=(d['raw_node']-s['node_mean'])/s['node_std']
        d['pair_x']=(d['raw_pair']-s['pair_mean'])/s['pair_std']
        z,c=m.forward(d);zs.append(z);hs.append(c[2]);es.append(c[-1])
    z=np.zeros_like(zs[0])
    for g in d['groups']:
        sl=g['slice'];z[sl]=np.log(np.maximum(np.mean([softmax(a[sl][None,:])[0] for a in zs],axis=0),1e-12))
    # Fixed random projections preserve both frozen encoder spaces; no label fit.
    rng=np.random.default_rng(217)
    h=np.tanh(np.concatenate(hs,axis=1)@rng.normal(0,1/8,(64,8)))
    e=np.tanh(np.concatenate(es,axis=1)@rng.normal(0,1/10,(96,16)))
    gs=d['groups'];starts=np.array([g['slice'].start for g in gs]);nb=len(d['buses']);n=len(gs)
    global_bus=np.full((n,max([len(g['buses'])+1 for g in gs],default=1)),nb,int)
    bm={int(b):i for i,b in enumerate(d['buses'])}
    for i,g in enumerate(gs):
        global_bus[i,:len(g['buses'])]=[bm[b] for b in g['buses']]
    di=np.array([g['device'] for g in gs],int);ii,jj=np.triu_indices(n,1)
    fs=[a['features'] for a in d['d']['nodes']]
    samevol=np.array([fs[di[i]]['voltage_kv']==fs[di[j]]['voltage_kv'] for i,j in zip(ii,jj)])
    ii=ii[samevol];jj=jj[samevol]
    peer=d['peer_mask'][di[ii],di[jj]] if len(ii) else np.array([],bool)
    pair=np.c_[h[di[ii]]*h[di[jj]],np.abs(h[di[ii]]-h[di[jj]])]
    return {'d':d,'z':z,'starts':starts,'global':global_bus,'h':h,'edge':e,'di':di,'ii':ii,'jj':jj,'peer':peer,'pair':pair,
            'base':np.array([assign(z,g) for g in gs]),'hard':np.array([g['hard'] for g in gs],bool),
            'sizes':np.array([len(g['buses'])+1 for g in gs]),'nb':nb}

def stats(s,a,y):
    base=s['base'];wrong=a!=y;old=base!=y;h=s['hard'];none=s['sizes']-1
    return Counter(devices=len(y),errors=int(wrong.sum()),hard_devices=int(h.sum()),hard_errors=int(wrong[h].sum()),
        corrected=int((old&~wrong&h).sum()),introduced=int((~old&wrong&h).sum()),
        hard_abstentions=int(((a==none)&h).sum()),tp=int(((a==y)&(y!=none)).sum()),
        fp=int(((a!=y)&(a!=none)).sum()),fn=int(((a!=y)&(y!=none)).sum()))

def metric(c):
    c=dict(c);c['hard_error_rate']=c['hard_errors']/max(c['hard_devices'],1)
    c['bus_precision']=c['tp']/max(c['tp']+c['fp'],1);c['bus_recall']=c['tp']/max(c['tp']+c['fn'],1)
    c['bus_f1']=2*c['tp']/max(2*c['tp']+c['fp']+c['fn'],1);return c
