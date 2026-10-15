"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .topology_structure_rules import normalize

def coupler_number(name):
    s=normalize(name)
    m=re.search(r'#?([0-9]+|VIII|VII|III|VI|IV|II|IX|V|I|一|二|三|四|五|六|七|八|九|十)号?母联',s)
    if not m:return None
    v=m[1];mapping={'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,'IX':9,'一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8,'九':9,'十':10}
    return int(v) if v.isdigit() else mapping[v]
