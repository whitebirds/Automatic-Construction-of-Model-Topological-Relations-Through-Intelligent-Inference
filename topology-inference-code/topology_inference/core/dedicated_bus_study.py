"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from . import run, candidate_experiment as ex
from .whole_station_study import make_input, attach_labels
from .character_hierarchical_network import encode, vocabulary
from .expanded_structure_study import descriptor
from .dedicated_bus_network import BusNet, softmax

def inputs(document,types,vocab=None):
    run.USE_BUS=True;b=make_input(document,types);ns=b['d']['nodes'];fs=[n['features'] for n in ns];n=len(ns)
    ds=np.array([i for i,f in enumerate(fs) if f['type_code']=='408'],int);buses=np.array([i for i,f in enumerate(fs) if f['type_code']=='410'],int);busmap={int(i):k for k,i in enumerate(buses)}
    lookup={tuple(sorted((int(i),int(j)))):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    keys=[]
    for i,f in enumerate(fs):
        _,key,_=descriptor(f);keys.append((f['voltage_kv'],key) if key else ('singleton',i))
    peer=np.array([[a==bb for bb in keys] for a in keys],bool)
    bm=np.array([[b['voltage_ok'][lookup[tuple(sorted((int(di),int(bi))))]] for di in ds] for bi in buses],bool).reshape(len(buses),len(ds))
    hard={g['ds_index'] for g in ex.group_inputs(b['d'],b)};left=[];right=[];pair=[];gs=[];allidx=[]
    for di in ds:
        candidates=[int(bi) for bi in buses if b['voltage_ok'][lookup[tuple(sorted((int(di),int(bi))))]]]
        idx=np.array([lookup[tuple(sorted((int(di),bi)))] for bi in candidates],int);start=len(left)
        for bi,k in zip(candidates,idx):left.append(int(di));right.append(busmap[bi]);pair.append(np.r_[b['x'][k,-9:],0.])
        left.append(int(di));right.append(len(buses));pair.append(np.r_[np.zeros(9),1.])
        gs.append({'device':int(di),'indices':idx,'buses':candidates,'slice':slice(start,len(left)),'hard':int(di) in hard,'weight':4. if int(di) in hard else 1.})
        allidx.extend([lookup[tuple(sorted((int(di),int(bi))))] for bi in buses])
    data={'d':b['d'],'base':b,'raw_node':b['tokens'],'raw_pair':np.array(pair,np.float32).reshape(-1,10),'peer_mask':peer,'devices':ds,'buses':buses,'bus_mask':bm,
          'left':np.array(left,int),'right':np.array(right,int),'groups':gs,'all_bus_pair_indices':np.array(allidx,int)}
    if vocab is not None:data['text_ids']=encode(b['d'],vocab)
    return data

def targets(data,document):
    b=data['base'];attach_labels(b,document)
    for g in data['groups']:
        y=b['y'][g['indices']];assert y.sum()<=1
        di=g['device'];allidx=np.flatnonzero(((b['i']==di)&np.isin(b['j'],data['buses']))|((b['j']==di)&np.isin(b['i'],data['buses'])))
        assert b['y'][allidx].sum()==y.sum(),'True target omitted from voltage candidate pool'
        g['target']=int(np.argmax(y)) if y.sum() else len(y)

def assign(z,g):
    a=z[g['slice']];best=np.flatnonzero(np.isclose(a,a.max(),rtol=0,atol=1e-9));return int(best[0]) if len(best)==1 else len(a)-1

def measures(data,z):
    c=Counter();ce=0.
    for g in data['groups']:
        pred=assign(z,g);true=g['target'];none=len(g['indices']);ok=pred==true
        c['devices']+=1;c['errors']+=int(not ok);c['hard_devices']+=int(g['hard']);c['hard_errors']+=int(g['hard'] and not ok)
        c['true_bus_devices']+=int(true!=none);c['wrong_bus_devices']+=int(true!=none and not ok);c['false_bus_devices']+=int(true==none and pred!=none)
        c['hard_abstentions']+=int(g['hard'] and pred==none);ce-=np.log(max(softmax(z[g['slice']][None,:])[0][true],1e-12))
    c['ce_sum']=float(ce);return c

def key(c):return (c['hard_errors']/max(c['hard_devices'],1),c['errors']/max(c['devices'],1),c['ce_sum']/max(c['devices'],1))
