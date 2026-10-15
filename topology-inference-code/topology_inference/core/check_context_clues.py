"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .naming import parse_name

def circuit_key(name):
    # Normalize explicit ordinal syntax, not Roman characters inside dispatch codes.
    s=re.sub(r'\d+(?:\.\d+)?\s*(?:kV|KV|kv|千伏)','',name)
    s=re.sub(r'\s+','',s)
    s=re.sub(r'#?(\d+)号?(?=主变|母联)',lambda m:'#'+m[1],s)
    # Section bays contain two sides. A group is a clue source, not a direct edge.
    if '分段' in s or '母分' in s:
        return 'section:'+re.split(r'分段|母分',s,maxsplit=1)[0]
    if '母联' in s:
        return 'coupler:'+s.split('母联',1)[0]+'母联'
    m=re.match(r'(#\d+主变)',s)
    if m:return 'transformer:'+m[1]
    stem=re.split(r'正母|副母|旁母|旁路母线|线路|母线侧|开关|断路器|闸刀|刀闸|接地|压变|电压互感器|避雷',s,maxsplit=1)[0]
    stem=re.sub(r'(?:线路|线)$','',stem)
    if len(stem)<3 or stem in ('母线','正母','副母'):return None
    return 'exact_name_stem:'+stem
