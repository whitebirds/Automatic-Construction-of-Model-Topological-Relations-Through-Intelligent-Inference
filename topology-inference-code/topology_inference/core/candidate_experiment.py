"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .check_context_clues import circuit_key, parse_name

CONTEXT_NAMES=['same_group_peer_count','peer_type_407','peer_type_408','peer_type_409','peer_type_415','peer_type_416','peer_type_463','peer_type_515',
 'peer_section_matches','peer_section_conflicts','peer_section_unknown','peer_family_matches','peer_family_conflicts','peer_family_unknown',
 'peer_same_family_section_matches','peer_same_family_section_conflicts','has_multiple_peer_sections']

def group_inputs(d,b):
    """Labels/nd are not accessed here. IDs are output references only."""
    ns=d['nodes'];fs=[n['features'] for n in ns]
    ps=[parse_name(f['name'],f['type_code']) for f in fs]
    bypair={(int(i),int(j)):k for k,(i,j) in enumerate(zip(b['i'],b['j']))}
    keys=[(f['voltage_kv'],circuit_key(f['name'])) for f in fs];result=[]
    for di,f in enumerate(fs):
        p=ps[di]
        if f['type_code']!='408' or len(p['bus_family_candidates'])!=1 or p['bus_section_candidates']:continue
        buses=[j for j,g in enumerate(fs) if g['type_code']=='410' and g['voltage_kv']==f['voltage_kv'] and ps[j]['bus_family_candidates']==p['bus_family_candidates']]
        if len(buses)<2:continue
        peers=[j for j,g in enumerate(fs) if j!=di and g['type_code']!='410' and keys[j]==keys[di]] if keys[di][1] else []
        cx=[]
        for bi in buses:
            sec=ps[bi]['bus_section_candidates'];fam=p['bus_family_candidates']
            matches=[j for j in peers if len(sec)==1 and ps[j]['bus_section_candidates']==sec]
            conflicts=[j for j in peers if len(sec)==1 and len(ps[j]['bus_section_candidates'])==1 and ps[j]['bus_section_candidates']!=sec]
            fm=[j for j in peers if ps[j]['bus_family_candidates']==fam]
            fc=[j for j in peers if len(ps[j]['bus_family_candidates'])==1 and ps[j]['bus_family_candidates']!=fam]
            vals={s for j in fm for s in ps[j]['bus_section_candidates']}
            cx.append([len(peers)]+[sum(fs[j]['type_code']==t for j in peers) for t in ['407','408','409','415','416','463','515']]+
                      [len(matches),len(conflicts),sum(not ps[j]['bus_section_candidates'] for j in peers),len(fm),len(fc),
                       sum(not ps[j]['bus_family_candidates'] for j in peers),len(set(matches)&set(fm)),len(set(conflicts)&set(fm)),int(len(vals)>1)])
        result.append({'ds_index':di,'bus_indices':buses,'pair_indices':np.array([bypair[tuple(sorted((di,j)))] for j in buses]),
                       'context':np.array(cx,np.float32),'peer_indices':peers})
    return result
