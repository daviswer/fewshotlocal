import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class Block(nn.Module):
    def __init__(self, insize, outsize):
        super(Block, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(insize, outsize, kernel_size=3, padding=1),
            nn.BatchNorm2d(outsize)
        )
        
    def forward(self, inp):
        return self.layers(inp)

# The 4-layer feature extractor
class PROTO(nn.Module):
    def __init__(self, w):
        super(PROTO, self).__init__()
        self.process = nn.Sequential(
            Block(3,w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Block(w,w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Block(w,w),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            Block(w,w)
        )
        
    def forward(self, inp):
        return self.process(inp)

    
#-----------PREDICTORS


# Basic prototype prediction
def predict(centroids, query):
    # centroids and query must be broadcastable!
    distmat = torch.sum((centroids-query)**2,-1).neg().view(-1, centroids.size(-2))
    return F.log_softmax(distmat, dim=-1)


# Unfolded prediction
class ufPredict(nn.Module):
    def __init__(self, trainshot):
        super(ufPredict, self).__init__()
        self.trainshot = trainshot
        
    def forward(self, inp, way):
        support = inp[:way*self.trainshot].view(way,self.trainshot,inp.size(-1))
        query = inp[way*self.trainshot:].unsqueeze(1) # b 1 d
        centroids = torch.mean(support, 1).unsqueeze(0) # 1 w d
        return predict(centroids, query)


# Batch folded prediction
class fPredict(nn.Module):
    def __init__(self, shot):
        super(fPredict, self).__init__()
        self.shot = shot
        
    def forward(self, inp, way):
        inp = inp.view(way, self.shot, inp.size(-1)) # w s d
        centroids = inp.mean(1).unsqueeze(0).unsqueeze(0) # 1 1 w d
        support = inp.unsqueeze(2) # w s 1 d
        eye = torch.eye(way).cuda().unsqueeze(1).unsqueeze(-1) # w 1 w 1
        rescaler = (eye/(self.shot-1) + 1)
        # Fold out the contribution of each point from its corresponding centroid
        centroids = (centroids - support*eye/self.shot) * rescaler # w s w d
        return predict(centroids, support)
        

#-----------FEATURE EXPANDERS AND POOLING OPERATIONS


# Covariance pooling
def covapool(inp, _, __):
    out = (inp.unsqueeze(1)*inp.unsqueeze(2)).view(inp.size(0), (inp.size(1))**2, inp.size(2)*inp.size(3))
    out = out.mean(-1)
    return out.sign().float()*(out.abs()+.00001).sqrt() # signed sqrt normalization
    
    
# Folded few-shot localization, with/without covariance pooling
def fL(inp, m, way, covariancing):
    b = inp.size(0)
    # Create folded bcentroids, mask off inputs
    masks = torch.cat([m, 1-m], dim=1).view(b, 1, 2, m.size(2), m.size(3)) # B 1 2 10 10
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01 # B 1 2
    bcentroids = (inp.unsqueeze(2)*masks).view(b, inp.size(1), 2, -1).sum(-1)/bsize # B 64 2
    bcentroids = (bcentroids.sum(0).unsqueeze(0)-bcentroids).unsqueeze(-1).unsqueeze(-1)/(b-1) # B 64 2 1 1
    masks = torch.sum((inp.unsqueeze(2)-bcentroids)**2, 1).neg().unsqueeze(1) # B 1 2 10 10
    masks = F.softmax(masks, dim=2)
    # Perform fore/back separation and appropriate expansion
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
    out = (inp.unsqueeze(2)*masks).view(b, inp.size(1), 2, -1) # B 64 2 100
    if covariancing:
        out = (out/bsize.unsqueeze(-1).sqrt()) # B 64 2 100
        out = (out[:,:,0].unsqueeze(1)*out[:,:,1].unsqueeze(2)).view(b, -1, out.size(-1)).sum(-1) # B 64*64
        return out.sign().float()*(out.abs()+.00001).sqrt() # signed sqrt normalization
    else: 
        out = out.sum(-1)/bsize # B 64 2
        return out.view(out.size(0), out.size(1)*2)


# Unfolded few-shot localization, with/without covariance pooling
def ufL(inp, m, way, bshot, covariancing):
    # Create folded bcentroids, mask off inputs
    m = m[:way*bshot]
    masks = torch.cat([m, 1-m], dim=1).view(m.size(0), 1, 2, m.size(2), m.size(3)) # B 1 2 10 10
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01 # B 1 2
    bcentroids = (inp[:way*bshot].unsqueeze(2)*masks).view(way*bshot, inp.size(1), 2, -1).sum(-1)/bsize # B 64 2
    bcentroids = bcentroids.mean(0).view(1, inp.size(1), 2, 1, 1) # 1 64 2 1 1
    masks = torch.sum((inp[way*bshot:].unsqueeze(2)-bcentroids)**2, 1).neg().unsqueeze(1) # B 1 2 10 10
    masks = F.softmax(masks, dim=2)
    # Perform fore/back separation and bilinear expansion
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
    out = (inp[way*bshot:].unsqueeze(2)*masks).view(inp.size(0)-way*bshot, inp.size(1), 2, -1) # B 64 2 100
    if covariancing:
        out = (out/bsize.unsqueeze(-1).sqrt()) # B 64 2 100
        out = (out[:,:,0].unsqueeze(1)*out[:,:,1].unsqueeze(2)).view(out.size(0), -1, out.size(-1)).sum(-1) # B 64*64
        return out.sign().float()*(out.abs()+.00001).sqrt() # signed sqrt normalization
    else:
        out = out.sum(-1)/bsize # B 64 2
        return out.view(out.size(0), out.size(1)*2)
    
    
# Parametric localizer
class pL(nn.Module):
    def __init__(self, w):
        super(pL, self).__init__()
        self.sm = nn.Softmax(dim=2)
        self.centroids = Parameter(torch.randn(1,w,2,1,1))
        
    def forward(self, inp, _, __):
        masks = torch.sum((inp.unsqueeze(2)-self.centroids)**2, 1).neg().unsqueeze(1) # B 1 2 10 10
        masks = self.sm(masks)
        # Perform fore/back separation 
        bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
        out = (inp.unsqueeze(2)*masks).view(inp.size(0), inp.size(1), 2, -1).sum(-1)/bsize # B 64 2
        return out.view(out.size(0), out.size(1)*2)
    
    
# Parametric localizer with covariance pooling
class pCL(nn.Module):
    def __init__(self, w):
        super(pCL, self).__init__()
        self.sm = nn.Softmax(dim=2)
        self.centroids = Parameter(torch.randn(1,w,2,1,1))
        
    def forward(self, inp, _, __):
        b = inp.size(0)
        masks = torch.sum((inp.unsqueeze(2)-self.centroids)**2, 1).neg().unsqueeze(1) # B 1 2 10 10
        masks = self.sm(masks)
        # Perform fore/back separation and bilinear expansion
        bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
        out = (inp.unsqueeze(2)*masks).view(b, inp.size(1), 2, -1) # B 64 2 100
        out = (out/bsize.unsqueeze(-1).sqrt()) # B 64 2 100
        out = (out[:,:,0].unsqueeze(1)*out[:,:,1].unsqueeze(2)).view(b, -1, out.size(-1)).sum(-1) # B 64*64
        return out.sign().float()*(out.abs()+.00001).sqrt() # signed sqrt normalization
    
    
# Baseline prototypical network
class avgpool(nn.Module):
    def __init__(self):
        super(avgpool, self).__init__()
        self.pool = nn.AvgPool2d(10)
        
    def forward(self, inp, _, __):
        return self.pool(inp).view(inp.size(0), inp.size(1))
        

#-----------FEATURE EXPANDERS AND POOLING OPERATIONS, TEST TIME ONLY


#         (Because the batches are much larger, foreground/background 
#         prediction and feature expansion must now be done separately)


# Foreground/background representations for few-shot localizers
def fbpredict(inp, m):
    b = inp.size(0)
    # Create folded bcentroids, mask off inputs
    masks = torch.cat([m, 1-m], dim=1).view(b, 1, 2, m.size(2), m.size(3)) # B 1 2 10 10
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01 # B 1 2
    return (inp.unsqueeze(2)*masks).view(b, inp.size(1), 2, -1).sum(-1)/bsize # B 64 2


# Few-shot localization
def fsL(inp, fbvectors, _):
    bcentroids = fbvectors.view(1, *fbvectors.size(), 1, 1) # 1 64 2 1 1
    masks = torch.sum((inp.unsqueeze(2)-bcentroids)**2, 1).neg().unsqueeze(1) # B 64 2 10 10
    masks = F.softmax(masks, dim=2)
    # Perform fore/back separation and concatenate
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
    out = (inp.unsqueeze(2)*masks).view(inp.size(0), inp.size(1), 2, -1) # B 64 2 100
    out = out.sum(-1)/bsize # B 64 2
    return out.view(out.size(0), out.size(1)*2)


# Few-shot localization with covariance pooling
def fsCL(inp, fbvectors, _):
    bcentroids = fbvectors.view(1, *fbvectors.size(), 1, 1) # 1 64 2 1 1
    masks = torch.sum((inp.unsqueeze(2)-bcentroids)**2, 1).neg().unsqueeze(1) # B 64 2 10 10
    masks = F.softmax(masks, dim=2)
    # Perform fore/back separation and concatenate
    bsize = masks.view(*masks.size()[:-2], -1).sum(-1)+.01
    out = (inp.unsqueeze(2)*masks).view(inp.size(0), inp.size(1), 2, -1) # B 64 2 100
    out = (out/bsize.unsqueeze(-1).sqrt()) # B 64 2 100
    out = (out[:,:,0].unsqueeze(1)*out[:,:,1].unsqueeze(2)).view(inp.size(0), -1, out.size(-1)).sum(-1) # B 64*64
    return out.sign().float()*(out.abs()+.00001).sqrt() # signed sqrt normalization

        
#-----------THE ACTUAL NETWORK
    
    
class Network(nn.Module):
    def __init__(self, w, folding, cova, local, proto, shots):
        super(Network, self).__init__()
        self.encode = PROTO(w)
        self.shots = shots
        # Numerical codes here correspond to the codes used in the ablation study figure
        if not local:
            # 001 and 101, and 000 and 100
            self.postprocess = covapool if cova else avgpool()
        elif proto:
            if folding:
                # 111 and 101
                self.postprocess = lambda x,y,z: fL(x,y,z, cova) # folded Localizer
            else:
                # 011 and 001
                self.postprocess = lambda x,y,z: ufL(x,y,z, shots[0], cova) # unfolded Localizer
        else:
            # 021 and 121, and 020 and 120
            self.postprocess = pCL(w) if cova else pL(w) # parametric Localizer
            
        if folding:
            self.predict = fPredict(shots[0]) # Folded prediction
        else:
            self.predict = ufPredict(shots[-2]) # Unfolded prediction
    
    
    def forward(self, inp, masks):
        assert inp.size(0)%sum(self.shots) == 0, "Error: batch size does not match given shot values."
        way = inp.size(0)//sum(self.shots)
        out = self.encode(inp)
        out = self.postprocess(out, masks, way)
        out = self.predict(out, way)
        return out
        

    
    
    
    
    
