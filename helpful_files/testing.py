import numpy as np
import torch
import pylab as pl
from time import sleep
from IPython import display
from copy import deepcopy
from PIL import Image
from torch.utils.data import Sampler
from helpful_files.networks import fbpredict, predict



def load_transform(path, boxdict, transform, masking):
    # Load the image
    with open(path, 'rb') as f:
        p = Image.open(f)
        p = p.convert('RGB')
    t = transform(p)
    # Load the bounding boxes
    m = t
    if masking:
        allmasks = np.zeros((10,10))
        boxes = boxdict[path]
        for box in boxes:
            mask = np.zeros((10,10))
            xmin = box[0]
            xmax = box[2]-.0000001 # Prevents overflow when xmax is exactly 10
            ymin = box[1]
            ymax = box[3]-.0000001
            xmin_int = int(xmin)
            xmax_int = int(xmax)+1
            ymin_int = int(ymin)
            ymax_int = int(ymax)+1
            mask[ymin_int:ymax_int, xmin_int:xmax_int] = 1
            # Fade out the left and right edges of the mask
            mask[:, xmin_int] *= 1 - (xmin - xmin_int)
            mask[:, xmax_int-1] *= 1 - (xmax_int - xmax)
            # Fade out the top and bottom edges of the mask
            mask[ymin_int,:] *= 1 - (ymin - ymin_int)
            mask[ymax_int-1,:] *= 1 - (ymax_int - ymax)
            # Take the union of the previous and current masks
            allmasks = 1 - (1-allmasks)*(1-mask) 
        m = torch.FloatTensor(allmasks).unsqueeze(0)
    return [t, m]


class OrderedSampler(Sampler):
    def __init__(self, data_source, bsize):
        iddict = dict()
        for i,(_,cat) in enumerate(data_source.imgs):
            if cat in iddict:
                iddict[cat].append(i)
            else:
                iddict[cat] = [i]
        self.iddict = iddict
        self.bsize = bsize
        
    def __iter__(self):
        trackdict = deepcopy(self.iddict)
        for key in trackdict:
            np.random.shuffle(trackdict[key])
        for key in trackdict:
            idlist = trackdict[key]
            size = len(idlist)
            if size <= self.bsize:
                yield idlist
            else:
                for i in range(len(idlist)//self.bsize):
                    yield idlist[i*self.bsize:(i+1)*self.bsize]
                if size%self.bsize != 0:
                    yield idlist[-(len(idlist)%self.bsize):]


# Accumulate foreground/background prototypes
def accumulateFB(models, loader, way, network_width, ngiven, bsize):
    catindex = 0
    lastcat = -1
    esize = len(models)
    fbcentroids = torch.zeros(esize, network_width, 2).cuda()
    progress = torch.zeros(1, way)
    for i, ((inp, mask), cat) in enumerate(loader):
        catindex = cat[0]

        # Moving to another category
        if catindex != lastcat:
            lastcat = catindex 
            slack = ngiven[catindex]
            progress[0, lastcat] = 1
            # Plot progress
            display.clear_output(wait=True)
            pl.figure(figsize=(20,1))
            pl.imshow(progress.numpy(), cmap='Greys')
            pl.title("Accumulating foreground/background prototypes:")
            pl.xticks([])
            pl.yticks([])
            pl.show()
            sleep(.01)

        # Determine how many images to use
        if slack > 0:
            if slack > bsize:
                inp = inp.cuda()
                mask = mask.cuda()
            else:
                inp = inp[:slack].cuda()
                mask = mask[:slack].cuda()
            
            # Continue accumulating
            with torch.no_grad():
                for j in range(esize):
                    out = models[j](inp) # b 64 10 10
                    fbcentroids[j] += fbpredict(out, mask).sum(0)
            slack -= bsize
            
    return fbcentroids/float(sum(ngiven)) # esize 64 2


# Accumulate category prototypes
def accumulate(models, loader, expanders, bcentroids, way, d):
    esize = len(models)
    centroids = torch.zeros(esize, way, d).cuda()
    catindex = 0
    lastcat = -1
    count = 0
    running = torch.zeros(esize, d).cuda()
    counts = [0]*way
    progress = torch.zeros(1, way)
    for i, ((inp,_), cat) in enumerate(loader):
        catindex = cat[0]

        # Moving to another category
        if catindex != lastcat: 
            if i != 0:
                for j in range(esize):
                    centroids[j, lastcat] = running[j]/count # Write the values
                counts[lastcat] = count
            lastcat = catindex # Record the current category
            count = 0 # Reset divisor
            running.zero_() # Reset accumulator
            progress[0, lastcat] = 1
            # Plot progress
            display.clear_output(wait=True)
            pl.figure(figsize=(20,1))
            pl.imshow(progress.numpy(), cmap='Greys')
            pl.title("Accumulating category prototypes:")
            pl.xticks([])
            pl.yticks([])
            pl.show()
            sleep(.01)

        # Continue accumulating
        inp = inp.cuda()
        with torch.no_grad():
            for j in range(esize):
                out = models[j](inp) # b 64 10 10
                out = expanders[j](out, bcentroids[j], None) # b d
                running[j] += out.sum(0) # Accumulate prototypes
        count += inp.size(0) # Accumulate the divisor

    # Record last category
    for j in range(esize):
        centroids[j, catindex] = running[j]/count
        counts[catindex] = count
    
    return centroids, counts


# Evaluation on query images
def score(k, centroids, bcentroids, models, loader, expanders, way):
    esize = len(models)
    right = [0]*esize
    allright = [0]*esize
    perclassacc = np.array([[0.]*way for _ in range(esize)])
    catindex = 0
    lastcat = -1
    count = 0
    allcount = 0
    progress = torch.zeros(1, way)
    for i, ((inp,_), cat) in enumerate(loader):
        catindex = cat[0]
        if catindex != lastcat: # We're about to move to another category
            # Write the values
            if i!= 0:
                allcount += count
                for j in range(esize):
                    allright[j] += right[j] 
                    perclassacc[j, lastcat] = right[j]/count
            lastcat = catindex # Record the current category
            count = 0 # Reset divisor
            right = [0]*esize # Reset accumulator
            progress[0, lastcat] = 1
            # Plot progress
            display.clear_output(wait=True)
            pl.figure(figsize=(20,1))
            pl.imshow(progress.numpy(), cmap='Greys')
            pl.title("Accumulating accuracy scores:")
            pl.xticks([])
            pl.yticks([])
            pl.show()
            sleep(.01)

        # Predict
        inp = inp.cuda()
        targ = cat.cuda()
        with torch.no_grad():
            for j in range(esize):
                out = models[j](inp)
                out = expanders[j](out, bcentroids[j], None)
                out = predict(centroids[j].unsqueeze(0), out.unsqueeze(1))
                _, pred = out.topk(k, 1, True, True)
                pred = pred.t()
                right[j] += pred.eq(targ.view(1, -1).expand_as(pred)).contiguous()[:k].view(-1).sum(0, keepdim=True).float().item()
        count += inp.size(0)

    # Record last category
    allcount += count
    for j in range(esize):
        allright[j] += right[j]
        perclassacc[j, catindex] = right[j]/count

    # Final reporting / recording
    allacc = [r/allcount for r in allright]
    
    return allacc, np.mean(perclassacc, axis=0), np.mean(perclassacc, axis=1)















