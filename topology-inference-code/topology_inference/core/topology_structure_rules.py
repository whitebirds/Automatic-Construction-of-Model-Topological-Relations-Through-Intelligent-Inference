"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .check_context_clues import circuit_key
VERSION = "equipment-structure-public-v1"

OPTIONAL_TERMINAL_ROLES={'line_pt','line_object','line_la','line_es'}

BUS_DS={'main_bus_ds','aux_bus_ds','bypass_bus_ds'}

def normalize(s):return re.sub(r'\s+','',unicodedata.normalize('NFKC',s).upper())

def equipment_role(f):
    t=f['type_code'];s=normalize(f['name'])
    if t=='408':
        roles=[]
        if '正母' in s:roles.append('main_bus_ds')
        if '副母' in s:roles.append('aux_bus_ds')
        if '旁母' in s or '旁路母线' in s:roles.append('bypass_bus_ds')
        if re.search(r'线路(?:闸刀|刀闸|隔离开关)',s):roles.append('line_ds')
        return roles[0] if len(roles)==1 else None
    if t=='409':
        roles=[]
        if re.search(r'(?:开关母线侧|母线侧|开关正母侧|开关副母侧).*?(?:接地|地刀)',s):roles.append('breaker_bus_es')
        if re.search(r'开关线路侧.*?(?:接地|地刀)',s):roles.append('breaker_line_es')
        if re.search(r'线路(?:接地|地刀)',s):roles.append('line_es')
        return roles[0] if len(roles)==1 else None
    if t=='407':return 'breaker'
    if t=='415':return 'line_object'
    if t=='463' and ('线路' in s or '线压变' in s):return 'line_pt'
    if t=='515' and ('线路' in s or '线避雷' in s):return 'line_la'
    return None

def forbidden_roles(a,b):
    return ((a=='line_es' and b in BUS_DS) or (b=='line_es' and a in BUS_DS)
            or frozenset((a,b))==frozenset(('line_ds','breaker_bus_es')))

def input_layout(b):
    """No labels consulted; grouping preserves exact dispatch suffixes."""
    fs=[n['features'] for n in b['d']['nodes']];roles=[equipment_role(f) for f in fs]
    forbidden=np.array([forbidden_roles(roles[i],roles[j]) for i,j in zip(b['i'],b['j'])],bool)
    bus_groups=[]
    for i,f in enumerate(fs):
        if f['type_code']=='408':
            idx=np.flatnonzero(((b['i']==i)&np.array([fs[j]['type_code']=='410' for j in b['j']]))|
                               ((b['j']==i)&np.array([fs[j]['type_code']=='410' for j in b['i']])))
            bus_groups.append((i,idx))
    raw_groups=defaultdict(list)
    for i,f in enumerate(fs):
        if f['type_code']=='410':continue
        name=normalize(f['name'])
        if any(w in name for w in ('主变','母联','分段','母分','中性','电容','所用')):continue
        stripped=re.sub(r'\d+(?:\.\d+)?KV','',name)
        codes=re.findall(r'(?<![A-Z0-9])[0-9][A-Z][A-Z0-9]*',stripped)
        key=('dispatch:'+codes[0]) if len(codes)==1 else circuit_key(name)
        if key:raw_groups[(f['voltage_kv'],key)].append(i)
    pair={tuple(sorted((int(i),int(j)))):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    groups=[];skipped=Counter()
    for key,indices in raw_groups.items():
        rr=[roles[i] for i in indices]
        if 'line_ds' not in rr:skipped['no_line_disconnector_anchor']+=1;continue
        if None in rr:skipped['unresolved_role']+=1;continue
        if len(set(rr))!=len(rr):skipped['duplicate_role']+=1;continue
        if len(rr)<3:skipped['too_small']+=1;continue
        mapping=dict(zip(rr,indices));pairs=[];edge_roles=[]
        for k,a in enumerate(sorted(mapping)):
            for bb in sorted(mapping)[k+1:]:
                pairs.append(pair[tuple(sorted((mapping[a],mapping[bb])))]);edge_roles.append((a,bb))
        groups.append({'key':key,'role_nodes':mapping,'indices':np.array(pairs,int),'edge_roles':edge_roles})
    return {'roles':roles,'forbidden':forbidden,'bus_groups':bus_groups,'groups':groups,'skipped':dict(skipped)}

def enforce(b,pred,scores,layout=None):
    layout=layout if layout is not None else input_layout(b)
    out=np.asarray(pred,bool).copy();out[layout['forbidden']]=False
    for device,idx in layout['bus_groups']:
        chosen=idx[out[idx]]
        if len(chosen)>1:
            best=chosen[scores[chosen]==scores[chosen].max()];out[chosen]=False
            if len(best)==1:out[best[0]]=True
    return out

def connected(roles,edges):
    if not roles:return False
    seen={next(iter(roles))}
    while True:
        nxt=seen|{v for e in edges if set(e)&seen for v in e}
        if nxt==seen:return len(seen)==len(roles)
        seen=nxt

def learn_templates(cache,files):
    """Only this function reads historical training labels."""
    templates={};excluded=[]
    for filename in files:
        b=cache[filename]
        for g in b['layout']['groups']:
            roles=tuple(sorted(g['role_nodes']));edges=tuple(sorted(e for e,k in zip(g['edge_roles'],g['indices']) if b['y'][k]))
            reason=None
            if any(forbidden_roles(*e) for e in edges):reason='contradicts_confirmed_role_rule'
            elif not connected(set(roles),edges):reason='disconnected_attribute_group'
            if reason:excluded.append({'file':filename,'key':g['key'],'reason':reason});continue
            key=(roles,edges)
            if key not in templates:templates[key]={'roles':list(roles),'edges':[list(e) for e in edges],'count':0,'stations':set()}
            templates[key]['count']+=1;templates[key]['stations'].add(filename)
    return [{**v,'stations':sorted(v['stations'])} for v in templates.values()],excluded

def project_template(template,query_roles):
    """Contract optional degree-2 vertices, delete optional leaves; never branch hubs."""
    source=set(template['roles']);target=set(query_roles)
    if not target<=source:return None
    missing=source-target
    if not missing<=OPTIONAL_TERMINAL_ROLES:return None
    edges={tuple(sorted(e)) for e in template['edges']}
    for role in sorted(missing):
        neighbors={v for e in edges if role in e for v in e if v!=role}
        if len(neighbors)>2:return None
        edges={e for e in edges if role not in e}
        if len(neighbors)==2:edges.add(tuple(sorted(neighbors)))
    if any(forbidden_roles(*e) for e in edges):return None
    if not connected(target,edges):return None
    return tuple(sorted(edges)),len(missing)

def candidate_templates(group,templates):
    choices={}
    for t in templates:
        projected=project_template(t,group['role_nodes'])
        if projected is None:continue
        edges,removed=projected
        if edges not in choices:choices[edges]={'edges':[list(e) for e in edges],'weight':0.,'exact_weight':0.,'stations':set(),'exact_stations':set(),'contracted_stations':set()}
        c=choices[edges];c['weight']+=t['count']/(4**removed);c['stations'].update(t['stations'])
        if not removed:c['exact_weight']+=t['count']
        c['contracted_stations' if removed else 'exact_stations'].update(t['stations'])
    result=[]
    for v in choices.values():
        if len(v['stations'])<3:continue
        result.append({**v,'stations':sorted(v['stations']),'exact_stations':sorted(v['exact_stations']),
                       'contracted_stations':sorted(v['contracted_stations']),
                       'mask':np.array([list(e) in v['edges'] for e in group['edge_roles']],bool)})
    return result

def apply_templates(b,pred,scores,threshold,prior_strength,layout=None,choices=None,allow_contract=True):
    layout=layout if layout is not None else b['layout'];out=pred.copy();used=[]
    for gi,g in enumerate(layout['groups']):
        candidates=choices[gi]
        if not allow_contract:candidates=[c for c in candidates if len(c['exact_stations'])>=3]
        if not candidates:continue
        objective=[]
        weight_key='weight' if allow_contract else 'exact_weight'
        total=sum(c[weight_key] for c in candidates)
        for c in candidates:
            objective.append(float(np.sum((scores[g['indices']]-threshold)*c['mask']))+prior_strength*math.log(c[weight_key]/total))
        top=max(objective);winners=[i for i,v in enumerate(objective) if abs(v-top)<1e-9]
        if len(winners)!=1:continue
        c=candidates[winners[0]];before=out[g['indices']].copy();out[g['indices']]=c['mask']
        used.append({'group':gi,'key':g['key'],'role_nodes':g['role_nodes'],'edges':c['edges'],
                     'changed_pairs':int(np.sum(before!=c['mask'])),'supporting_stations':c['stations'],
                     'exact_support_stations':len(c['exact_stations']),'contracted_support_stations':len(c['contracted_stations'])})
    return enforce(b,out,scores,layout),used

def structural_checks():
    roles=['line_pt','line_object','line_la','line_es','line_ds']
    t={'roles':roles,'edges':[list(e) for e in zip(roles[:-1],roles[1:])]}
    assert set(project_template(t,['line_pt','line_es','line_ds'])[0])=={tuple(sorted(e)) for e in [('line_pt','line_es'),('line_es','line_ds')]}
    assert project_template({'roles':['line_ds','breaker','breaker_bus_es'],'edges':[['line_ds','breaker'],['breaker','breaker_bus_es']]},['line_ds','breaker_bus_es']) is None
    assert forbidden_roles('main_bus_ds','line_es') and forbidden_roles('line_ds','breaker_bus_es')
    assert not forbidden_roles('line_ds','line_es') and not forbidden_roles('line_ds','breaker_line_es')
    return {'optional_chain_contraction':True,'cannot_contract_breaker':True,'role_prohibitions':True,'two_ground_neighbors_allowed':True}
