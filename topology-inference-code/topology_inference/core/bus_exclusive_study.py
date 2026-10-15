"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .repeat_context_attention import tokens
from .check_context_clues import circuit_key, parse_name

def groups_from_inputs(b,types):
    nodes=b['d']['nodes'];fs=[n['features'] for n in nodes];tok=tokens(b['d'],types);mean=tok.mean(0)
    pair={tuple(sorted((int(i),int(j)))):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    keys=[(f['voltage_kv'],circuit_key(f['name'])) for f in fs];result=[]
    for di,f in enumerate(fs):
        if f['type_code']!='408':continue
        buses=[i for i,g in enumerate(fs) if g['type_code']=='410' and b['voltage_ok'][pair[tuple(sorted((di,i)))]]]
        indices=np.array([pair[tuple(sorted((di,i)))] for i in buses],int)
        peers=[i for i,g in enumerate(fs) if i!=di and g['type_code']!='410' and keys[i]==keys[di]] if keys[di][1] else []
        pm=tok[peers].mean(0) if peers else np.zeros(tok.shape[1],np.float32)
        rr=[];hx=[]
        for bi in buses:
            rr.append(np.r_[tok[di],tok[bi],pm,mean,len(peers)/10,len(buses)/10,0])
            hx.append(np.r_[tok[di],tok[bi],pm,b['x'][pair[tuple(sorted((di,bi)))],-9:]])
        rr.append(np.r_[tok[di],np.zeros_like(pm),pm,mean,len(peers)/10,len(buses)/10,1])
        result.append({'device':di,'id':nodes[di]['node_id'],'bus_indices':buses,'indices':indices,
                       'context':np.array(rr,np.float32),'history_x':np.array(hx,np.float32).reshape(len(buses),3*tok.shape[1]+9),
                       'voltage':f['voltage_kv'],'name_section_known':bool(parse_name(f['name'],'408')['bus_section_candidates'])})
    b['signature']=np.r_[mean,tok.std(0),np.log1p([len(nodes)]+[sum(f['type_code']==t for f in fs) for t in types])]
    return result
