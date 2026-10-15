"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .station_dialect_study import codevec
from .dedicated_bus_study import targets
from .check_context_clues import parse_name
from .dedicated_bus_network import softmax
from . import run

TYPES=['407','408','409','415','416','463','515','405','other']

def graph_vectors(s,adj):
    d=s['d'];ns=d['d']['nodes'];n=len(ns);typ=np.zeros((n,len(TYPES)))
    for i,node in enumerate(ns):
        t=node['features']['type_code'];typ[i,TYPES.index(t) if t in TYPES else -1]=1
    rows=[]
    for g in d['groups']:
        i=g['device'];peers=np.flatnonzero(d['peer_mask'][i]);peers=np.array([j for j in peers if ns[j]['features']['type_code']!='410'])
        a=adj[np.ix_(peers,peers)];r=int(np.flatnonzero(peers==i)[0]);one=a[r]@typ[peers];two=(a[r]@a)@typ[peers]
        rows.append(np.r_[np.log1p(one),np.log1p(two)]/3)
    return np.array(rows)

def enrich(s,stage):
    d=s['d'];b=d['base'];ns=d['d']['nodes'];fs=[n['features'] for n in ns];p=[parse_name(f['name'],f['type_code']) for f in fs]
    model,mean,std,threshold=stage;z=model.predict((b['x']-mean)/std);adj=np.zeros((len(ns),len(ns)))
    w=run.sigmoid(z-threshold)*b['voltage_ok'];adj[b['i'],b['j']]=w;adj[b['j'],b['i']]=w
    code=np.array([codevec(f['name']) for f in fs]);root=[]
    for g in d['groups']:
        i=g['device'];peers=np.flatnonzero(d['peer_mask'][i]);name=fs[i]['name']
        counts=[sum(fs[j]['type_code']==t for j in peers) for t in TYPES[:-1]]
        root.append(np.r_[s['h'][i],code[i],s['h'][peers].mean(0),np.log1p(counts),[word in name for word in ('线路','主变','母联','分段','正母','副母')]])
    bus=[]
    for bi in d['buses']:
        pp=p[bi];bus.append(np.r_[s['h'][bi],[k in pp['bus_section_candidates'] for k in range(1,9)],len(pp['bus_section_candidates'])==0])
    s.update(root=np.array(root),qgraph=graph_vectors(s,adj),bus=np.array(bus).reshape(-1,17),parsed=p,
             sig=np.r_[s['h'].mean(0),code.mean(0),np.log1p([sum(f['type_code']==t and f['voltage_kv']==v for f in fs) for t in TYPES[:-1] for v in (10,35,110,220,500)])])
    return s

def history(s):
    b=s['d']['base'];n=len(s['d']['d']['nodes']);a=np.zeros((n,n));a[b['i'],b['j']]=b['y'];a[b['j'],b['i']]=b['y']
    s['rgraph']=graph_vectors(s,a)

def build(query,fn,bank,train,sigstd,*,physical_ids=None,reference_count=8,equipment_count=8):
    """Self/physical-station exclusion applies also to training queries."""
    if reference_count < 1 or equipment_count < 1:
        raise ValueError('Reference and equipment counts must be positive.')
    groups = physical_ids if physical_ids is not None else {f: f for f in [fn, *train]}
    if any(f not in groups for f in [fn, *train]):
        raise ValueError('Missing physical station group identity.')
    eligible=[f for f in train if groups[f] != groups[fn]]
    order=sorted(eligible,key=lambda f:float(np.mean(((query['sig']-bank[f]['sig'])/sigstd)**2)))[:reference_count]
    tasks=[];nodes=query['d']['d']['nodes'];buslookup={b:i for i,b in enumerate(query['d']['buses'])}
    for gi,g in enumerate(query['d']['groups']):
        if not g['hard']:continue
        di=g['device'];family=query['parsed'][di]['bus_family_candidates'];vol=nodes[di]['features']['voltage_kv'];refs=[]
        qb=np.array([query['bus'][buslookup[b]] for b in g['buses']]+[np.zeros(17)])
        for f in order:
            ref=bank[f];rn=ref['d']['d']['nodes'];rbl={b:i for i,b in enumerate(ref['d']['buses'])};valid=[]
            for j,rg in enumerate(ref['d']['groups']):
                rd=rg['device'];y=ref['y'][j]
                if y==len(rg['buses']) or rn[rd]['features']['voltage_kv']!=vol:continue
                if ref['parsed'][rd]['bus_family_candidates']!=family:continue
                valid.append(j)
            if not valid:continue
            # Attribute-only local prefilter; preserve all roots tied at cutoff.
            dist=np.mean(abs(ref['root'][valid]-query['root'][gi]),axis=1);cut=np.partition(dist,min(equipment_count-1,len(dist)-1))[min(equipment_count-1,len(dist)-1)]
            keep=np.array(valid)[dist<=cut+1e-12];rr=ref['root'][keep];qr=np.broadcast_to(query['root'][gi],rr.shape)
            ax=np.c_[abs(qr-rr),qr*rr]
            graph=abs(query['qgraph'][gi]-ref['rgraph'][keep])
            rb=np.array([ref['bus'][rbl[ref['d']['groups'][j]['buses'][ref['y'][j]]]] for j in keep])
            qv=np.broadcast_to(qb[None,:,:],(len(keep),len(qb),17));rv=np.broadcast_to(rb[:,None,:],qv.shape)
            none=np.zeros((len(keep),len(qb),1));none[:,-1]=1
            tx=np.concatenate([abs(qv-rv),qv*rv,none],axis=2)
            # Fixed map prioritizes explicit section agreement, allowing unknowns.
            sect=np.sum(abs(qv[:,:,8:16]-rv[:,:,8:16]),axis=2)
            fixed_t=-4*sect-.2*np.mean(abs(qv-rv),axis=2)-8*none[:,:,0]
            refs.append({'file':f,'root_indices':keep,'station':np.abs((query['sig']-ref['sig'])/sigstd),
                'attr':ax,'structure':np.c_[ax,graph],'transfer':tx,'fixed_align':-2*np.mean(abs(qr-rr),axis=1)-np.mean(graph,axis=1),'fixed_transfer':fixed_t})
        tasks.append({'group':gi,'base':np.exp(query['z'][g['slice']]),'refs':refs})
    return tasks

class Head:
    def __init__(self,dim,width,r):self.p=[r.normal(0,1/np.sqrt(dim),(dim,width)),np.zeros(width),r.normal(0,.01,width)]
    def forward(self,x):
        h=np.tanh(x@self.p[0]+self.p[1]);return h@self.p[2],(x,h)
    def back(self,dz,c):
        x,h=c;dh=dz[:,None]*self.p[2]*(1-h*h);return [x.T@dh,dh.sum(0),h.T@dz]

class MemoryNet:
    def __init__(self,dims,seed,mode):
        r=np.random.default_rng(seed);self.mode=mode;self.heads=[Head(d,16,r) for d in dims];self.null=np.array([1.])
        self.m=[np.zeros_like(p) for p in self.params()];self.v=[np.zeros_like(p) for p in self.params()];self.t=0
    def params(self):return [p for h in self.heads for p in h.p]+[self.null]
    def forward(self,t,fixed=False,memory_only=False):
        refs=t['refs'];base=t['base']
        if not refs:return base,None
        sx=np.array([r['station'] for r in refs]);rs,sc=self.heads[0].forward(sx);rc=[];vals=[]
        for r in refs:
            ax=r[self.mode];az,ac=self.heads[1].forward(ax);tx=r['transfer'];tz,tc=self.heads[2].forward(tx.reshape(-1,tx.shape[-1]))
            if fixed:az=r['fixed_align'];tz=r['fixed_transfer'].ravel()
            aw=softmax(az[None,:])[0];tr=softmax(tz.reshape(tx.shape[:2]));vals.append(aw@tr);rc.append((aw,tr,ac,tc))
        if fixed:rs=-np.mean(sx*sx,axis=1)
        null=-1e9 if memory_only else (self.null[0] if not fixed else 0.)
        sw=softmax(np.r_[null,rs][None,:])[0];vv=np.vstack([base,vals]);out=sw@vv
        return out,(sc,rc,sw,vv)
    def loss_grad(self,t,y,memory_only=False):
        pr,c=self.forward(t,memory_only=memory_only);gs=[np.zeros_like(p) for p in self.params()];loss=-np.log(max(pr[y],1e-12))
        if c is None:return loss,gs
        sc,rc,sw,vv=c;den=max(pr[y],1e-12);ds=-sw*(vv[:,y]-pr[y])/den;gs[:3]=self.heads[0].back(ds[1:],sc);gs[-1][0]=ds[0]
        for k,(aw,tr,ac,tc) in enumerate(rc):
            da=-sw[k+1]*aw*(tr[:,y]-vv[k+1,y])/den
            dt=sw[k+1]*aw[:,None]*tr*tr[:,y,None]/den;dt[:,y]-=sw[k+1]*aw*tr[:,y]/den
            for j,g in enumerate(self.heads[1].back(da,ac)):gs[3+j]+=g
            for j,g in enumerate(self.heads[2].back(dt.ravel(),tc)):gs[6+j]+=g
        return loss,gs
    def step(self,t,y,reference_loss=0.):
        loss,gs=self.loss_grad(t,y);self.t+=1
        if reference_loss:
            aux,ag=self.loss_grad(t,y,True);loss+=reference_loss*aux;gs=[g+reference_loss*a for g,a in zip(gs,ag)]
        for i,(p,g) in enumerate(zip(self.params(),gs)):
            g=np.clip(g,-5,5)+1e-4*p;self.m[i]=.9*self.m[i]+.1*g;self.v[i]=.999*self.v[i]+.001*g*g
            p-=.001*(self.m[i]/(1-.9**self.t))/(np.sqrt(self.v[i]/(1-.999**self.t))+1e-8)
        return loss
    def restore(self,ps):
        for p,a in zip(self.params(),ps):p[:]=a

def evaluate(s,tasks,net,strength,fixed=False,details=False):
    a=s['base'].copy();traces=[]
    for t in tasks:
        p,c=net.forward(t,fixed);a[t['group']]=int(np.argmax((1-strength)*t['base']+strength*p))
        if details and c is not None:
            k=int(np.argmax(c[2][1:]));r=t['refs'][k];root=int(np.argmax(c[1][k][0]));traces.append({'group':t['group'],'reference':r['file'],'reference_root':int(r['root_indices'][root]),'station_weight':float(c[2][k+1]),'null_weight':float(c[2][0])})
    return a,traces
