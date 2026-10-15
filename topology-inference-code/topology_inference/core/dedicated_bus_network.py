"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .character_hierarchical_network import CharacterCNN

def softmax(z):
    a=np.exp(z-z.max(-1,keepdims=True));return a/a.sum(-1,keepdims=True)

def attention(qx,kx,mask,wq,wk,wv):
    q=qx@wq;k=kx@wk;v=kx@wv
    a=softmax(q@k.T/np.sqrt(q.shape[1])+np.where(mask,0.,-1e4))*mask
    a/=np.maximum(a.sum(1,keepdims=True),1e-12)
    return a@v,(qx,kx,q,k,v,a)

def back_attention(do,c,wq,wk,wv):
    qx,kx,q,k,v,a=c;da=do@v.T;ds=a*(da-(da*a).sum(1,keepdims=True));scale=np.sqrt(q.shape[1])
    dq=ds@k/scale;dk=ds.T@q/scale;dv=a.T@do
    return dq@wq.T,dk@wk.T+dv@wv.T,[qx.T@dq,kx.T@dk,kx.T@dv]

class BusNet:
    def __init__(self,node_dim,pair_dim,vocab_size,seed,mode='joint',width=32):
        self.mode=mode;self.width=width;r=np.random.default_rng(seed)
        def w(a,b):return (r.normal(size=(a,b))/np.sqrt(a)).astype(np.float32)
        self.p=[w(node_dim+48,width),np.zeros(width,np.float32)]+[w(width,width) for _ in range(6)]
        self.p +=[r.normal(0,.1,width).astype(np.float32),w(4*width+pair_dim,48),np.zeros(48,np.float32),w(48,1),np.zeros(1,np.float32)]
        self.char=CharacterCNN(vocab_size,seed+31);self.p+=self.char.p;self.m=[np.zeros_like(x) for x in self.p];self.v=[np.zeros_like(x) for x in self.p];self.t=0

    def forward(self,b):
        p=self.p;self.char.p=p[13:];text,cc=self.char.forward(b['text_ids'],(True,True));x=np.c_[b['node_x'],text];h=np.tanh(x@p[0]+p[1])
        if self.mode=='joint':
            local,lc=attention(h,h,b['peer_mask'],*p[2:5]);e=np.tanh(h+local)
            if len(b['buses']) and len(b['devices']):
                global_,gc=attention(h[b['buses']],e[b['devices']],b['bus_mask'],*p[5:8]);bh=np.tanh(h[b['buses']]+global_)
            else:gc=None;bh=h[b['buses']]
        else:lc=gc=None;e=h;bh=h[b['buses']]
        bh=np.vstack((bh,np.tanh(p[8])));left=e[b['left']];right=bh[b['right']]
        pair=np.c_[left,right,left*right,np.abs(left-right),b['pair_x']];hidden=np.tanh(pair@p[9]+p[10]);z=(hidden@p[11]+p[12]).ravel()
        return z,(cc,x,h,lc,e,gc,bh,left,right,pair,hidden)

    def loss_grad(self,b):
        z,c=self.forward(b);dz=np.zeros_like(z);loss=0.;weight=sum(g['weight'] for g in b['groups'])
        for g in b['groups']:
            sl=g['slice'];prob=softmax(z[sl][None,:])[0];factor=g['weight']/max(weight,1);loss-=factor*np.log(max(prob[g['target']],1e-12));dz[sl]=factor*prob;dz[sl.start+g['target']]-=factor
        cc,x,h,lc,e,gc,bh,left,right,pair,hidden=c;p=self.p;gr=[np.zeros_like(a) for a in p]
        gr[11]=hidden.T@dz[:,None];gr[12]=np.array([dz.sum()]);dhid=(dz[:,None]@p[11].T)*(1-hidden*hidden);gr[9]=pair.T@dhid;gr[10]=dhid.sum(0)
        dp=dhid@p[9].T;a,bb,m,d=np.split(dp[:,:4*self.width],4,axis=1);sign=np.sign(left-right)
        de=np.zeros_like(e);db=np.zeros_like(bh);np.add.at(de,b['left'],a+m*right+d*sign);np.add.at(db,b['right'],bb+m*left-d*sign)
        gr[8]=db[-1]*(1-bh[-1]*bh[-1]);dh=np.zeros_like(h)
        if self.mode=='joint':
            if gc is not None:
                dt=db[:-1]*(1-bh[:-1]*bh[:-1]);np.add.at(dh,b['buses'],dt)
                dq,dk,gg=back_attention(dt,gc,*p[5:8]);np.add.at(dh,b['buses'],dq);np.add.at(de,b['devices'],dk);gr[5:8]=gg
            else:np.add.at(dh,b['buses'],db[:-1])
            dt=de*(1-e*e);dh+=dt;dq,dk,gg=back_attention(dt,lc,*p[2:5]);dh+=dq+dk;gr[2:5]=gg
        else:dh+=de;np.add.at(dh,b['buses'],db[:-1])
        dt=dh*(1-h*h);gr[0]=x.T@dt;gr[1]=dt.sum(0);dx=dt@p[0].T;gr[13:]=self.char.backward(dx[:,-48:],cc)
        return float(loss),gr

    def step(self,b,lr=.0003):
        loss,gg=self.loss_grad(b);norm=np.sqrt(sum(float(np.sum(g*g)) for g in gg));scale=min(1.,5/max(norm,1e-9));self.t+=1
        for k,(p,g) in enumerate(zip(self.p,gg)):
            g=g*scale+1e-5*p;self.m[k]=.9*self.m[k]+.1*g;self.v[k]=.999*self.v[k]+.001*g*g
            p-=lr*(self.m[k]/(1-.9**self.t))/(np.sqrt(self.v[k]/(1-.999**self.t))+1e-8)
        return loss

def gradient_check():
    r=np.random.default_rng(5);n=5;b={'node_x':r.normal(size=(n,3)),'text_ids':[r.integers(1,15,(n,7)),r.integers(1,15,(n,6))],
        'peer_mask':np.ones((n,n),bool),'buses':np.array([3,4]),'devices':np.array([0,1]),'bus_mask':np.ones((2,2),bool),
        'left':np.array([0,0,0,1,1,1]),'right':np.array([0,1,2,0,1,2]),'pair_x':r.normal(size=(6,2)),
        'groups':[{'slice':slice(0,3),'target':1,'weight':4},{'slice':slice(3,6),'target':2,'weight':1}]}
    errors=[]
    for mode in ('independent','joint'):
        net=BusNet(3,2,15,7,mode,width=4);net.p=[p.astype(np.float64) for p in net.p];loss,gr=net.loss_grad(b)
        for k,p in enumerate(net.p):
            for _ in range(2):
                idx=np.unravel_index(r.integers(p.size),p.shape);old=p[idx];eps=1e-5;p[idx]=old+eps;a=net.loss_grad(b)[0];p[idx]=old-eps;bb=net.loss_grad(b)[0];p[idx]=old
                error=abs((a-bb)/(2*eps)-gr[k][idx]);assert error<2e-5,(mode,k,error);errors.append(error)
    return {'checks':len(errors),'max_error':max(errors)}
