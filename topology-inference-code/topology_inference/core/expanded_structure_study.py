"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .topology_structure_rules import normalize, equipment_role, input_layout, enforce, connected
from .audit_remaining_roles import coupler_number

NUM={'I':'1','II':'2','III':'3','IV':'4','V':'5','VI':'6','VII':'7','VIII':'8','IX':'9','一':'1','二':'2','三':'3','四':'4','五':'5','六':'6','七':'7','八':'8','九':'9','十':'10'}

PAT=r'(VIII|VII|III|VI|IV|II|IX|V|I|一|二|三|四|五|六|七|八|九|十|\d+)'

def canonical(name):
    s=normalize(name).split('.')[-1] if '.' in normalize(name) and not re.search(r'\d\.\d',name) else normalize(name)
    s=re.sub(r'\d+(?:\.\d+)?(?:KV|千伏)','',s)
    s=re.sub(PAT+r'(?=段|号?母联|号?主变)',lambda m:NUM.get(m[1],m[1]),s)
    s=re.sub(r'#?(\d+)号?(?=母联|主变)',r'#\1',s)
    return s

def descriptor(f):
    s=canonical(f['name']);t=f['type_code']
    if t=='410':return None,None,None
    for word,kind in [('母联','coupler'),('分段','section'),('母分','section'),('主变','transformer')]:
        if word in s:
            stem,tail=s.split(word,1)
            # Same type on different sides remains distinct; do not merge bus category and section.
            tail=re.sub(r'断路器|隔离开关|接地闸刀|接地刀闸|隔离刀闸|开关|闸刀|刀闸','',tail)
            return kind,kind+':'+stem, t+':'+tail
    if any(w in s for w in ('中性','电容','所用')):return None,None,None
    stem=re.split(r'正母|副母|旁母|线路|母线侧|开关|断路器|闸刀|刀闸|接地|压变|电压互感器|避雷',s,maxsplit=1)[0]
    stem=re.sub(r'线$','',stem)
    if len(stem)<3:return None,None,None
    return 'line','line:'+stem,equipment_role(f)

def groups(b):
    raw=defaultdict(list);ns=b['d']['nodes'];pairs={tuple(sorted((int(i),int(j)))):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    for i,n in enumerate(ns):
        kind,key,role=descriptor(n['features'])
        if key:raw[(n['features']['voltage_kv'],kind,key)].append((role,i))
    out=[]
    for key,items in raw.items():
        roles=[r for r,i in items]
        if None in roles or len(roles)!=len(set(roles)) or len(roles)<2:continue
        mapping=dict(items);rr=sorted(mapping);er=[(a,bb) for k,a in enumerate(rr) for bb in rr[k+1:]]
        out.append({'kind':key[1],'key':key,'roles':rr,'er':er,'idx':np.array([pairs[tuple(sorted((mapping[a],mapping[bb])))] for a,bb in er])})
    return out

def library(cache,files):
    ts={}
    for fn in files:
        b=cache[fn]
        for g in b['extended']:
            ee=tuple(e for e,k in zip(g['er'],g['idx']) if b['y'][k])
            if not connected(set(g['roles']),ee) or any(b['layout']['forbidden'][k] for k in g['idx'] if b['y'][k]):continue
            key=(g['kind'],tuple(g['roles']),ee)
            if key not in ts:ts[key]={'kind':g['kind'],'roles':g['roles'],'edges':ee,'stations':set()}
            ts[key]['stations'].add(fn)
    return [{**t,'stations':sorted(t['stations'])} for t in ts.values()]

def choices(g,ts):
    cc={}
    for t in ts:
        if t['kind']!=g['kind'] or not set(g['roles'])<=set(t['roles']):continue
        missing=set(t['roles'])-set(g['roles']);ee={tuple(e) for e in t['edges']}
        # Optional earthing/terminal devices only; never contract a breaker or branch hub.
        if any(r not in {'line_pt','line_object','line_la','line_es','breaker_line_es','breaker_bus_es'} and not r.startswith('409:') for r in missing):continue
        valid=True
        for r in sorted(missing):
            neighbors={v for e in ee if r in e for v in e if v!=r}
            if len(neighbors)>2:valid=False;break
            ee={e for e in ee if r not in e}
            if len(neighbors)==2:ee.add(tuple(sorted(neighbors)))
        if not valid or not connected(set(g['roles']),ee):continue
        key=tuple(sorted(ee));cc.setdefault(key,set()).update(t['stations'])
    return [{'mask':np.array([e in ee for e in g['er']]),'support':len(stations),'stations':sorted(stations)} for ee,stations in cc.items() if len(stations)>=3]

def guard(b,p):
    ns=b['d']['nodes'];nums=[coupler_number(n['features']['name']) for n in ns];out=p.copy()
    for k in np.flatnonzero(out):
        a,bb=nums[b['i'][k]],nums[b['j'][k]]
        if a is not None and bb is not None and a!=bb:out[k]=False
    return out

def decode(b,p,z,threshold,strength,cc):
    out=guard(b,p)
    if strength>=0:
        for g,cs in zip(b['extended'],cc):
            cs=[c for c in cs if not np.any(c['mask']&b['layout']['forbidden'][g['idx']])]
            if not cs:continue
            values=[float(np.sum((z[g['idx']]-threshold)*c['mask']))+strength*np.log(c['support']/sum(x['support'] for x in cs)) for c in cs]
            best=np.flatnonzero(np.isclose(values,max(values),rtol=0,atol=1e-9))
            if len(best)==1:out[g['idx']]=cs[best[0]]['mask']
    return guard(b,enforce(b,out,z,b['layout']))

def count(b,p,scope):
    fs=[n['features']['type_code'] for n in b['d']['nodes']]
    mask=np.ones(len(p),bool) if scope=='all' else np.array(['-'.join(sorted((fs[i],fs[j])))==scope for i,j in zip(b['i'],b['j'])])
    y=b['y'].astype(bool);return {'tp':int((p&y&mask).sum()),'fp':int((p&~y&mask).sum()),'fn':int((~p&y&mask).sum())}

def f1(c):return 2*c['tp']/max(1,2*c['tp']+c['fp']+c['fn'])
