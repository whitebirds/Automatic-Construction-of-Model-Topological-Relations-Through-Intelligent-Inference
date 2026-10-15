"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from . import run

class StationNet:
    def __init__(self, pairdim, tokdim, seed, global_context=True, width=16):
        self.global_context=global_context
        self.width=width
        r=np.random.default_rng(seed)
        def w(a,b): return (r.normal(size=(a,b))/np.sqrt(a)).astype(np.float32)
        self.p=[w(tokdim,width),np.zeros(width,np.float32)]
        # Two attention blocks: Q,K,V,O, global-mean projection, bias.
        for _ in range(2):
            self.p += [w(width,width) for _ in range(5)]+[np.zeros(width,np.float32)]
        dim=pairdim+3*width
        self.p += [w(dim,32),np.zeros(32,np.float32),w(32,16),np.zeros(16,np.float32),
                   np.zeros((16,1),np.float32),np.zeros(1,np.float32),w(width,1),np.zeros(1,np.float32)]
        self.m=[np.zeros_like(p) for p in self.p];self.v=[np.zeros_like(p) for p in self.p];self.t=0

    def forward(self, tokens, x, ii, jj, base, bias):
        p=self.p;h=np.tanh(tokens@p[0]+p[1]);initial=h;blocks=[]
        for k in (2,8):
            q=h@p[k];key=h@p[k+1];val=h@p[k+2]
            if self.global_context:
                z=q@key.T/np.sqrt(self.width)+bias
                a=np.exp(z-z.max(1,keepdims=True));a/=a.sum(1,keepdims=True)
                pool=np.broadcast_to(h.mean(0),h.shape)
            else:
                a=np.eye(len(h),dtype=np.float32);pool=h
            av=a@val
            out=np.tanh(h+av@p[k+3]+pool@p[k+4]+p[k+5])
            blocks.append((h,q,key,val,a,av,pool,out));h=out
        emb=np.c_[h[ii]+h[jj],np.abs(h[ii]-h[jj]),h[ii]*h[jj]]
        xx=np.c_[x,emb];f=np.maximum(xx@p[14]+p[15],0);g=np.maximum(f@p[16]+p[17],0)
        scores=base+(g@p[18]+p[19]).ravel()
        degree=(h@p[20]+p[21]).ravel()
        return scores,degree,(tokens,initial,blocks,h,xx,f,g,ii,jj)

    def loss_grad(self, tokens,x,ii,jj,base,bias,y,groups,target_degree):
        scores,degree,c=self.forward(tokens,x,ii,jj,base,bias)
        loss=float(np.mean(np.logaddexp(0,scores)-y*scores))
        dz=(run.sigmoid(scores)-y)/len(y)
        if groups:
            for idx,truth in groups:
                z=scores[idx];prob=np.exp(z-z.max());prob/=prob.sum()
                loss-=.15*float((truth*np.log(np.maximum(prob,1e-12))).sum())/len(groups)
                dz[idx]+=.15*(prob-truth)/len(groups)
        err=degree-target_degree;loss+=.02*float(np.mean(err*err));dd=.04*err/len(err)
        return loss,self.backward(c,dz,dd)

    def backward(self,c,dz,dd):
        tokens,initial,blocks,h,xx,f,g,ii,jj=c;p=self.p
        grads=[np.zeros_like(q) for q in p];dz=dz[:,None]
        dg=(dz@p[18].T)*(g>0);df=(dg@p[16].T)*(f>0)
        grads[14:20]=[xx.T@df,df.sum(0),f.T@dg,dg.sum(0),g.T@dz,dz.sum(0)]
        de=(df@p[14].T)[:,-3*self.width:]
        a,b,v=np.split(de,3,axis=1);sign=np.sign(h[ii]-h[jj])
        dh=np.zeros_like(h)
        np.add.at(dh,ii,a+b*sign+v*h[jj]);np.add.at(dh,jj,a-b*sign+v*h[ii])
        grads[20]=h.T@dd[:,None];grads[21]=np.array([dd.sum()],dtype=h.dtype);dh+=dd[:,None]@p[20].T
        for k,block in reversed(list(zip((2,8),blocks))):
            old,q,key,val,att,av,pool,out=block
            dt=dh*(1-out*out);dav=dt@p[k+3].T;dpool=dt@p[k+4].T
            grads[k+3]=av.T@dt;grads[k+4]=pool.T@dt;grads[k+5]=dt.sum(0)
            dv=att.T@dav;grads[k+2]=old.T@dv
            dh=dt+dv@p[k+2].T
            if self.global_context:
                da=dav@val.T;ds=att*(da-(da*att).sum(1,keepdims=True))
                dq=ds@key/np.sqrt(self.width);dk=ds.T@q/np.sqrt(self.width)
                grads[k]=old.T@dq;grads[k+1]=old.T@dk
                dh+=dq@p[k].T+dk@p[k+1].T+dpool.sum(0,keepdims=True)/len(old)
            else:
                dh+=dpool
        dt=dh*(1-initial*initial);grads[0]=tokens.T@dt;grads[1]=dt.sum(0)
        return grads

    def step(self,args,lr=.0005):
        loss,grads=self.loss_grad(*args)
        norm=np.sqrt(sum(float(np.sum(g*g)) for g in grads));scale=min(1.,5/max(norm,1e-9));self.t+=1
        for k,(p,g) in enumerate(zip(self.p,grads)):
            g=g*scale+1e-5*p
            self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            p-=lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
        return loss

    def predict(self,tokens,x,ii,jj,base,bias):
        # Entire station remains visible to the encoder for every pair.
        return self.forward(tokens,x,ii,jj,base,bias)[0]

def gradient_check():
    r=np.random.default_rng(31);n=6;ii,jj=np.triu_indices(n,1)
    model=StationNet(5,7,12)
    # Nonzero final layer is necessary to exercise encoder gradients.
    model.p=[p.astype(np.float64) for p in model.p];model.p[18][:]=r.normal(0,.1,model.p[18].shape)
    args=(r.normal(size=(n,7)),r.normal(size=(len(ii),5)),ii,jj,r.normal(size=len(ii)),
          r.normal(size=(n,n)),r.integers(0,2,size=len(ii)).astype(float),[(np.array([0,1]),np.array([1.,0.]))],r.normal(size=n))
    loss,grads=model.loss_grad(*args);errors=[]
    for k,p in enumerate(model.p):
        for _ in range(min(3,p.size)):
            idx=np.unravel_index(r.integers(p.size),p.shape);old=p[idx];eps=1e-5
            p[idx]=old+eps;a=model.loss_grad(*args)[0];p[idx]=old-eps;b=model.loss_grad(*args)[0];p[idx]=old
            numeric=(a-b)/(2*eps);error=abs(numeric-grads[k][idx]);errors.append(error)
            assert error < 2e-5,(k,idx,numeric,grads[k][idx])
    return {'checked':len(errors),'max_absolute_error':max(errors)}
