"""Generic naming conventions; not a universal naming standard."""
import re

ROMAN = {'I':1,'II':2,'III':3,'IV':4,'V':5,'VI':6,'VII':7,'VIII':8,
         'Ⅰ':1,'Ⅱ':2,'Ⅲ':3,'Ⅳ':4,'Ⅴ':5,'Ⅵ':6,'Ⅶ':7,'Ⅷ':8,
         '一':1,'二':2,'三':3,'四':4,'五':5,'六':6,'七':7,'八':8}

NUM = r'(VIII|VII|III|VI|IV|II|I|V|[ⅠⅡⅢⅣⅤⅥⅦⅧ一二三四五六七八1-8])'

def parse_name(name, type_code):
    # Never NFKC/replace the entire name: circuit Roman numerals stay intact.
    families = sorted({v for k,v in [('正母','main'),('副母','aux'),('旁母','bypass'),('旁路母线','bypass')] if k in name})
    spans = []
    patterns = [NUM + r'段', r'(?:正母|副母|旁母)' + NUM + r'(?=压变|闸刀|母线|线|$)']
    if '分段' in name:
        patterns.append(NUM + r'母(?=闸刀|刀闸)')
    for pattern in patterns:
        for m in re.finditer(pattern, name):
            # Avoid matching the suffix II of an unrelated alphabetic token.
            start = m.start(1)
            if start and name[start-1] in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ⅠⅡⅢⅣⅤⅥⅦⅧ':
                continue
            spans.append({'text':m.group(0),'start':m.start(),'end':m.end(),
                          'value':ROMAN.get(m.group(1), int(m.group(1)) if m.group(1).isdigit() else None)})
    sections = sorted({s['value'] for s in spans})
    roles = [v for k,v in [('母联','bus_coupler'),('分段','bus_section'),('母分','bus_section')] if k in name]
    sides = [v for k,v in [('线路侧','line'),('母线侧','bus'),('主变侧','transformer')] if k in name]
    # Broad exact tokens only, NOT a claimed authoritative dispatch-number field.
    tokens = re.findall(r'[0-9][A-Z][A-Z0-9]*(?:\s*(?:VIII|VII|III|VI|IV|II|I|V|[ⅠⅡⅢⅣⅤⅥⅦⅧ]))?', name)
    tokens = [x for x in tokens if not re.fullmatch(r'\d+V', x)]
    return {'bus_family_candidates':families, 'bus_section_candidates':sections,
            'bus_identity_status':'ambiguous' if len(families)>1 or len(sections)>1 else ('explicit_partial' if families or sections else 'unknown'),
            'bay_role_candidates':sorted(set(roles)), 'device_side_candidates':sorted(set(sides)),
            'dispatch_tokens_exact':tokens, 'section_evidence':spans}
