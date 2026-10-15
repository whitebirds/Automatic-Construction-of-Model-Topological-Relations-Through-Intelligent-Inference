"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
import json
from pathlib import Path

WORDS=['正母','副母','Ⅰ','Ⅱ','Ⅲ','分段','母联','旁路','线路','母线','主变','开关','闸刀','接地','压变','避雷','中性点','高压','低压','备用']

USE_BUS=False

BUS_FEATURE_NAMES=['family_both_known','family_equal','family_conflict','section_both_known','section_equal','section_conflict','bus_disconnector_pair','bus_pair_family_conflict','bus_pair_section_conflict']

def dump(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def sigmoid(x):return 1/(1+np.exp(-np.clip(x,-30,30)))

def clean(s):return re.sub(r'\s+','',s).upper().replace('ＫＶ','KV')

def prefix(a,b):
    k=0
    for x,y in zip(a,b):
        if x!=y:break
        k+=1
    return k

def codes(s):
    s=re.sub(r'\d+(?:\.\d+)?KV','',clean(s))
    return re.findall(r'(?<![A-Z0-9])[0-9][A-Z0-9]{2,}',s)

def assoc(a,b):
    def tokens(s):
        s=re.sub(r'\d+(?:\.\d+)?KV','',s)
        return set(re.findall(r'[0-9][A-Z][A-Z0-9]*(?:[ⅠⅡⅢⅣⅤⅥⅦⅧ])?',s))
    ca,cb=tokens(a),tokens(b)
    if ca and cb:return 1. if ca & cb else -1.
    ta,tb=re.search(r'#(\d+)主变',a),re.search(r'#(\d+)主变',b)
    if ta and tb:return 1. if ta[1]==tb[1] else -1.
    return 0.

def volts(f):
    v=f['winding_voltage_kv'] if f['type_code']=='416' else [f['voltage_kv']]
    return {float(x) for x in v if x is not None and float(x)>0}

def metrics(y,p):
    y=y.astype(bool);p=p.astype(bool)
    tp=int(np.sum(y&p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p))
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1),'edge_edits':fp+fn,'true_edges':int(y.sum()),'predicted_edges':int(p.sum())}

def best_threshold(y,s):
    # Exact all-pair validation search, one operating point for each method.
    order=np.argsort(-s,kind='stable');v=s[order];t=np.cumsum(y[order]);k=np.arange(1,len(y)+1)
    ends=np.r_[np.flatnonzero(v[:-1]!=v[1:]),len(v)-1]
    f=2*t[ends]/np.maximum(k[ends]+y.sum(),1);j=ends[int(np.argmax(f))]
    return float(v[j]),float(np.max(f))

def base_features(d,types,with_labels=True):
    nodes=d['nodes'];n=len(nodes);i,j=np.triu_indices(n,1)
    fs=[x['features'] for x in nodes];names=[clean(x['name']) for x in fs]
    tv=np.zeros((n,len(types)+1),np.float32)
    for k,f in enumerate(fs):tv[k,types.index(f['type_code']) if f['type_code'] in types else -1]=1
    w=np.array([[float(word.upper() in s) for word in WORDS] for s in names],np.float32)
    num=np.array([[np.log1p(f['voltage_kv'] or 0)/7,len(f['winding_voltage_kv'])/3,len(s)/40] for f,s in zip(fs,names)],np.float32)
    # Symmetric feature construction: swapping endpoints gives the same vector.
    x=np.concatenate([tv[i]+tv[j],np.abs(tv[i]-tv[j]),(tv[i,:,None]*tv[j,None,:]+tv[j,:,None]*tv[i,None,:]).reshape(len(i),-1),w[i]+w[j],np.abs(w[i]-w[j]),w[i]*w[j],num[i]+num[j],np.abs(num[i]-num[j])],axis=1)
    chars=[set(s) for s in names];bigrams=[{s[k:k+2] for k in range(len(s)-1)} for s in names]
    extra=[];ns=[];valid=[]
    vs=[volts(f) for f in fs]
    for a,b in zip(i,j):
        a=int(a);b=int(b);sa,sb=names[a],names[b]
        extra.append([len(chars[a]&chars[b])/max(len(chars[a]|chars[b]),1),len(bigrams[a]&bigrams[b])/max(len(bigrams[a]|bigrams[b]),1),prefix(sa,sb)/max(min(len(sa),len(sb)),1),float(sa==sb)])
        hint=assoc(sa,sb)
        # For bus links an explicit bus-side name is a weak association hint.
        if hint==0 and (fs[a]['type_code']=='410' or fs[b]['type_code']=='410'):
            matching=any(token in sa and token in sb for token in ['正母','副母'])
            hint=.5 if matching else 0.
        ns.append(hint)
        valid.append(not(vs[a] and vs[b] and vs[a].isdisjoint(vs[b])))
    x=np.concatenate([x,np.array(extra,np.float32)],axis=1).astype(np.float32)
    ids=[z['node_id'] for z in nodes];index={z:k for k,z in enumerate(ids)}
    labels=d['labels']['edges'] if with_labels else []
    edge={(min(index[e['source']],index[e['target']]),max(index[e['source']],index[e['target']])) for e in labels}
    assert all(a!=b for a,b in edge)
    y=np.array([(int(a),int(b)) in edge for a,b in zip(i,j)],np.float32)
    assert int(y.sum())==len(edge)==len(labels)
    # PT endpoint rule applies ONLY to explicit line PT under user convention.
    leaves={k for k,f in enumerate(fs) if f['type_code']=='463' and '线路' in names[k]}
    return {'x':x,'y':y,'name_hint':np.array(ns,np.float32),'voltage_ok':np.array(valid,bool),'i':i,'j':j,'leaves':leaves,'d':d}

def structure_mask(b,scores):
    mask=b['voltage_ok'].copy()
    # Select strongest valid incident edge. Selection sees predictions, not labels.
    for leaf in b['leaves']:
        idx=np.flatnonzero(((b['i']==leaf)|(b['j']==leaf)) & mask)
        if len(idx)>1:
            best=idx[np.argmax(scores[idx])];mask[idx]=False;mask[best]=True
    return mask

class MLP:
    def __init__(self,dim,seed):
        r=np.random.default_rng(seed)
        self.p=[(r.normal(size=(dim,48))*np.sqrt(2/dim)).astype(np.float32),np.zeros(48,np.float32),(r.normal(size=(48,24))*np.sqrt(2/48)).astype(np.float32),np.zeros(24,np.float32),(r.normal(size=(24,1))*np.sqrt(1/24)).astype(np.float32),np.zeros(1,np.float32)]
        self.m=[np.zeros_like(p) for p in self.p];self.v=[np.zeros_like(p) for p in self.p];self.t=0
    def forward(self,x):
        w,b,v,c,u,d=self.p;h=np.maximum(x@w+b,0);g=np.maximum(h@v+c,0);z=(g@u+d).ravel();return h,g,z
    def predict(self,x):return np.concatenate([self.forward(x[k:k+4096])[2] for k in range(0,len(x),4096)])
    def step(self,x,y,lr=.001):
        h,g,z=self.forward(x);p=sigmoid(z);dz=((p-y)/len(y))[:,None]
        dg=dz@self.p[4].T;dg[g<=0]=0;dh=dg@self.p[2].T;dh[h<=0]=0
        grads=[x.T@dh,dh.sum(0),h.T@dg,dg.sum(0),g.T@dz,dz.sum(0)]
        self.t+=1
        for k,(q,grad) in enumerate(zip(self.p,grads)):
            if k%2==0:grad+=1e-5*q
            self.m[k]=.9*self.m[k]+.1*grad;self.v[k]=.999*self.v[k]+.001*grad*grad
            q-=lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
        return float(np.mean(np.logaddexp(0,z)-y*z))

def features(d,types,with_labels=True):
    b=base_features(d,types,with_labels)
    if not USE_BUS:return b
    extra=[]
    for i,j in zip(b['i'],b['j']):
        a,c=[d['nodes'][k]['features'] for k in (i,j)]
        aa,cc=[f.get('bus_name_attributes',{}) for f in (a,c)]
        fs=[x.get('bus_family_candidates',[]) for x in (aa,cc)]
        ss=[x.get('bus_section_candidates',[]) for x in (aa,cc)]
        fk=all(len(x)==1 for x in fs);sk=all(len(x)==1 for x in ss)
        fe=fk and fs[0]==fs[1];se=sk and ss[0]==ss[1]
        bp={a['type_code'],c['type_code']}=={'410','408'}
        extra.append([fk,fe,fk and not fe,sk,se,sk and not se,bp,bp and fk and not fe,bp and sk and not se])
    b['x']=np.concatenate([b['x'],np.array(extra,dtype=np.float32)],axis=1)
    return b
