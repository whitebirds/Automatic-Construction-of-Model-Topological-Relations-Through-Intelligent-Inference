"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .whole_station_network import StationNet
from .check_context_clues import circuit_key
from . import run

def structure(b,scores,threshold):
    fs=[n['features'] for n in b['d']['nodes']];n=len(fs);groups={};membership=[]
    for i,f in enumerate(fs):
        key=circuit_key(f['name']) if f['type_code']!='410' else None
        key=(f['voltage_kv'],key) if key else ('singleton',i)
        if key not in groups:groups[key]=len(groups)
        membership.append(groups[key])
    unpool=np.zeros((n,len(groups)),np.float32);unpool[np.arange(n),membership]=1
    pool=unpool.T/unpool.sum(0)[:,None]
    local=np.eye(n,dtype=np.float32)
    # These are stage-one PREDICTIONS; no edges or target degrees are read.
    p=run.sigmoid(scores-threshold)
    for k,(i,j) in enumerate(zip(b['i'],b['j'])):
        if fs[i]['type_code']=='410' or fs[j]['type_code']=='410':continue
        if not b['voltage_ok'][k]:continue
        local[i,j]=local[j,i]=p[k]
    return {'local_bias':np.log(np.maximum(local,1e-5)), 'pool':pool,'unpool':unpool}

class HierNet(StationNet):
    def block(self,h,k,bias=None):
        p=self.p;q=h@p[k];key=h@p[k+1];v=h@p[k+2]
        z=q@key.T/np.sqrt(self.width)
        if bias is not None:z=z+bias
        a=np.exp(z-z.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
        av=a@v;mean=h if k==2 else np.broadcast_to(h.mean(0),h.shape)
        out=np.tanh(h+av@p[k+3]+mean@p[k+4]+p[k+5])
        return out,(h,q,key,v,a,av,mean,out)

    def forward(self,tokens,x,ii,jj,base,bias):
        p=self.p;initial=np.tanh(tokens@p[0]+p[1])
        local,bc1=self.block(initial,2,bias['local_bias'])
        pooled=bias['pool']@local
        global_h,bc2=self.block(pooled,8)
        h=np.tanh(local+bias['unpool']@global_h)
        emb=np.c_[h[ii]+h[jj],np.abs(h[ii]-h[jj]),h[ii]*h[jj]]
        xx=np.c_[x,emb];f=np.maximum(xx@p[14]+p[15],0);g=np.maximum(f@p[16]+p[17],0)
        scores=base+(g@p[18]+p[19]).ravel();degree=(h@p[20]+p[21]).ravel()
        return scores,degree,(tokens,initial,bc1,bc2,bias,h,xx,f,g,ii,jj)

    def backblock(self,dh,k,bc,grads):
        old,q,key,v,a,av,mean,out=bc;p=self.p;dt=dh*(1-out*out)
        dav=dt@p[k+3].T;dm=dt@p[k+4].T;dv=a.T@dav
        grads[k+3]=av.T@dt;grads[k+4]=mean.T@dt;grads[k+5]=dt.sum(0);grads[k+2]=old.T@dv
        da=dav@v.T;ds=a*(da-(da*a).sum(1,keepdims=True))
        dq=ds@key/np.sqrt(self.width);dk=ds.T@q/np.sqrt(self.width)
        grads[k]=old.T@dq;grads[k+1]=old.T@dk
        dmean=dm if k==2 else dm.sum(0,keepdims=True)/len(old)
        return dt+dv@p[k+2].T+dq@p[k].T+dk@p[k+1].T+dmean

    def backward(self,c,dz,dd):
        tokens,initial,bc1,bc2,bias,h,xx,f,g,ii,jj=c;p=self.p
        grads=[np.zeros_like(p) for p in self.p];dz=dz[:,None]
        dg=(dz@p[18].T)*(g>0);df=(dg@p[16].T)*(f>0)
        grads[14:20]=[xx.T@df,df.sum(0),f.T@dg,dg.sum(0),g.T@dz,dz.sum(0)]
        de=(df@p[14].T)[:,-3*self.width:];a,b,v=np.split(de,3,axis=1);sg=np.sign(h[ii]-h[jj]);dh=np.zeros_like(h)
        np.add.at(dh,ii,a+b*sg+v*h[jj]);np.add.at(dh,jj,a-b*sg+v*h[ii])
        grads[20]=h.T@dd[:,None];grads[21]=np.array([dd.sum()]);dh+=dd[:,None]@p[20].T
        # Optional auxiliary task gradient; ordinary models keep the old path.
        extra=getattr(self,'aux_embedding_gradient',None)
        if extra is not None:dh+=extra
        dt=dh*(1-h*h);dglobal=bias['unpool'].T@dt
        dpool=self.backblock(dglobal,8,bc2,grads)
        dlocal=dt+bias['pool'].T@dpool
        dh=self.backblock(dlocal,2,bc1,grads);dt=dh*(1-initial*initial)
        grads[0]=tokens.T@dt;grads[1]=dt.sum(0)
        self.input_gradient=dt@p[0].T
        return grads

def check_gradients():
    r=np.random.default_rng(19);n=6;ii,jj=np.triu_indices(n,1);model=HierNet(5,7,18)
    model.p=[p.astype(np.float64) for p in model.p];model.p[18][:]=r.normal(0,.1,model.p[18].shape)
    u=np.eye(3)[[0,0,1,1,2,2]];bias={'pool':u.T/2,'unpool':u,'local_bias':r.normal(size=(n,n))}
    args=(r.normal(size=(n,7)),r.normal(size=(len(ii),5)),ii,jj,r.normal(size=len(ii)),bias,
          r.integers(0,2,len(ii)).astype(float),[(np.array([0,1]),np.array([1.,0.]))],r.normal(size=n))
    loss,grads=model.loss_grad(*args);errors=[]
    for k,p in enumerate(model.p):
        for _ in range(min(3,p.size)):
            index=np.unravel_index(r.integers(p.size),p.shape);old=p[index];eps=1e-5
            p[index]=old+eps;a=model.loss_grad(*args)[0];p[index]=old-eps;b=model.loss_grad(*args)[0];p[index]=old
            error=abs((a-b)/(2*eps)-grads[k][index]);assert error<2e-5,(k,error);errors.append(error)
    return {'checks':len(errors),'max_error':max(errors)}
