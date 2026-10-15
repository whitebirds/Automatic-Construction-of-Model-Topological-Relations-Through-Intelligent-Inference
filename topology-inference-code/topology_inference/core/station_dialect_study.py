"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
import hashlib
from . import run

def codevec(name):
    cs=run.codes(unicodedata.normalize('NFKC',name));v=np.zeros(44)
    for c in cs:
        if c[-1].isdigit():v[int(c[-1])]+=1
        if len(c)>1:
            v[10+int(hashlib.sha256(c[-2:].encode()).hexdigest()[:8],16)%16]+=1
        for i in range(len(c)-1):v[26+int(hashlib.sha256(c[i:i+2].encode()).hexdigest()[:8],16)%16]+=1/max(len(c)-1,1)
    v[42]=bool(cs);v[43]=np.mean([len(c) for c in cs])/10 if cs else 0
    if cs:v[:42]/=len(cs)
    return v
