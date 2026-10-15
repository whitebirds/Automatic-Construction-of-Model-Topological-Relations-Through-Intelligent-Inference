"""Pure numerical implementation; no bundled data or saved models."""
import numpy as np
from collections import Counter, defaultdict
import re, math, copy, unicodedata
from .hierarchical_network import HierNet

def normalize(name):
    return re.sub(r'\s+','',unicodedata.normalize('NFKC',name)).upper()

def texts(d):
    names=[normalize(n['features']['name']) for n in d['nodes']]
    codes=[]
    for name in names:
        without_voltage=re.sub(r'\d+(?:\.\d+)?(?:KV|千伏)','',name)
        # Heuristic candidate extraction, not an authoritative dispatch field.
        pieces=re.findall(r'[A-Z0-9]+',without_voltage)
        codes.append('|'.join(pieces))
    return names,codes

def vocabulary(documents):
    chars=sorted({c for d in documents for channel in texts(d) for s in channel for c in s})
    return {'<PAD>':0,'<UNK>':1,'<BOS>':2,'<EOS>':3,**{c:i+4 for i,c in enumerate(chars)}}

def encode(d,vocab):
    result=[]
    for seqs,length in zip(texts(d),(64,32)):
        ids=np.zeros((len(seqs),length),np.int32)
        for i,s in enumerate(seqs):
            if not s:continue
            if len(s)>length-2:s=s[:(length-2)//2]+s[-((length-2)-(length-2)//2):]
            seq=[2]+[vocab.get(c,1) for c in s]+[3];ids[i,:len(seq)]=seq
        result.append(ids)
    return result

class CharacterCNN:
    def __init__(self,size,seed,embedding=8,filters=8):
        self.embedding=embedding;self.filters=filters;r=np.random.default_rng(seed)
        self.p=[r.normal(0,.15,(size,embedding)).astype(np.float32)];self.p[0][0]=0
        for channel in range(2):
            for width in (2,3,4):
                self.p.extend([(r.normal(size=(width*embedding,filters))/np.sqrt(width*embedding)).astype(np.float32),np.zeros(filters,np.float32)])

    def forward(self,inputs,enabled):
        n=len(inputs[0]);blocks=[];out=[]
        for channel,ids in enumerate(inputs):
            if not enabled[channel]:
                out.append(np.zeros((n,3*self.filters),dtype=self.p[0].dtype));continue
            emb=self.p[0][ids].copy();emb[ids==0]=0;lengths=(ids!=0).sum(1)
            for wi,width in enumerate((2,3,4)):
                k=1+6*channel+2*wi;steps=ids.shape[1]-width+1
                windows=np.concatenate([emb[:,j:j+steps] for j in range(width)],axis=2)
                h=np.tanh(windows@self.p[k]+self.p[k+1])
                valid=(np.arange(steps)[None,:]<np.maximum(lengths-width+1,1)[:,None])&(lengths[:,None]>0)
                arg=np.argmax(np.where(valid[:,:,None],h,-1e6),axis=1)
                pooled=h[np.arange(n)[:,None],arg,np.arange(self.filters)[None,:]]
                pooled=np.where(lengths[:,None]>0,pooled,0);out.append(pooled)
                blocks.append((channel,wi,k,ids,windows,h,arg,lengths,width))
        return np.concatenate(out,axis=1),blocks

    def backward(self,gradient,blocks):
        grads=[np.zeros_like(p) for p in self.p];n=len(gradient)
        for channel,wi,k,ids,windows,h,arg,lengths,width in blocks:
            part=gradient[:,(3*channel+wi)*self.filters:(3*channel+wi+1)*self.filters]
            dh=np.zeros_like(h);dh[np.arange(n)[:,None],arg,np.arange(self.filters)[None,:]]=part*(lengths[:,None]>0)
            dz=dh*(1-h*h);grads[k]=windows.reshape(-1,windows.shape[-1]).T@dz.reshape(-1,self.filters);grads[k+1]=dz.sum((0,1))
            dw=dz@self.p[k].T;de=np.zeros((n,ids.shape[1],self.embedding),dtype=dw.dtype);steps=windows.shape[1]
            for j in range(width):de[:,j:j+steps]+=dw[:,:,j*self.embedding:(j+1)*self.embedding]
            np.add.at(grads[0],ids.ravel(),de.reshape(-1,self.embedding))
        grads[0][0]=0
        return grads

class CharacterHierNet(HierNet):
    def __init__(self,pairdim,tokdim,vocab_size,seed,mode='name_code',warm=None):
        super().__init__(pairdim,tokdim+48,seed)
        self.mode=mode;self.enabled=(mode!='off',mode=='name_code');self.text=CharacterCNN(vocab_size,seed+23)
        if warm is not None:
            self.p=[w.copy() for w in warm]
            self.p[0]=np.concatenate([self.p[0],np.zeros((48,self.width),np.float32)],axis=0)
        self.p+=self.text.p;self.m=[np.zeros_like(w) for w in self.p];self.v=[np.zeros_like(w) for w in self.p];self.t=0

    def forward(self,tokens,x,ii,jj,base,bias):
        self.text.p=self.p[22:35]
        encoded,tc=self.text.forward(bias['text_ids'],self.enabled)
        scores,degree,hc=super().forward(np.c_[tokens,encoded],x,ii,jj,base,bias)
        return scores,degree,(hc,tc)

    def backward(self,c,dz,dd):
        hc,tc=c;grads=super().backward(hc,dz,dd)
        grads[22:35]=self.text.backward(self.input_gradient[:,-48:],tc)
        return grads

def check_gradients():
    rng=np.random.default_rng(201);n=5;ii,jj=np.triu_indices(n,1);model=CharacterHierNet(4,6,30,4)
    model.p=[w.astype(np.float64) for w in model.p];model.p[18][:]=rng.normal(0,.1,model.p[18].shape)
    u=np.eye(3)[[0,0,1,1,2]];bias={'pool':u.T/u.sum(0)[:,None],'unpool':u,'local_bias':rng.normal(size=(n,n)),
        'text_ids':[rng.integers(1,30,size=(n,8)),rng.integers(1,30,size=(n,6))]}
    args=(rng.normal(size=(n,6)),rng.normal(size=(len(ii),4)),ii,jj,rng.normal(size=len(ii)),bias,
          rng.integers(0,2,len(ii)).astype(float),[(np.array([0,1]),np.array([1.,0.]))],rng.normal(size=n))
    loss,grads=model.loss_grad(*args);errors=[]
    for k,p in enumerate(model.p):
        for _ in range(min(3,p.size)):
            idx=np.unravel_index(rng.integers(p.size),p.shape);old=p[idx];eps=1e-5
            p[idx]=old+eps;a=model.loss_grad(*args)[0];p[idx]=old-eps;b=model.loss_grad(*args)[0];p[idx]=old
            error=abs((a-b)/(2*eps)-grads[k][idx]);assert error<2e-5,(k,error);errors.append(error)
    return {'checks':len(errors),'max_error':max(errors),'character_gradient_norm':float(np.linalg.norm(grads[22]))}
