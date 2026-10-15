"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .check_context_clues import parse_name
from . import run

def tokens(d,types):
    rows=[]
    for n in d['nodes']:
        f=n['features'];p=parse_name(f['name'],f['type_code']);s=run.clean(f['name'])
        rows.append([float(f['type_code']==t) for t in types]+[float(w.upper() in s) for w in run.WORDS]+
                    [np.log1p(f['voltage_kv'] or 0)/7,len(f['winding_voltage_kv'])/3]+
                    [float(p['bus_family_candidates']==[v]) for v in ('main','aux','bypass')]+
                    [float(p['bus_section_candidates']==[v]) for v in range(1,9)])
    return np.array(rows,np.float32)
