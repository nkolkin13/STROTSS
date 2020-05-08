from collections import namedtuple
import random
import ssl

import torch.nn.functional as F
from torch.autograd import Variable
import torch
from torchvision import models
import numpy as np

ssl._create_default_https_context = ssl._create_unverified_context
use_random = True


class Vgg16_pt(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg16_pt, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.vgg_layers = vgg_pretrained_features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        for x in range(1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.inds = range(11)
        
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1)
        
        self.register_buffer('mu', mean)
        self.register_buffer('sig', std)

    def forward_base(self,X,rand):
        inds = self.inds

        x = (X-self.mu)/self.sig


        l2 = [X]
        for i in range(30):
            try:
                x =  self.vgg_layers[i].forward(x)#[:,:,1:-1,1:-1]
            except:
                pass
            if i in [1,3,6,8,11,13,15,22,29]:
                l2.append(x)

        return l2


    def forward(self, X, inds=[1,3,5,8,11], rand=True):


        inds = self.inds

        x = X
        l2 = self.forward_base(X,rand)
        out2 = l2

        return out2

    def forward_cat(self, X, r, inds=[1,3,5,8,11], rand=True,samps=100, forward_func=None):

        if not forward_func:
            forward_func = self.forward

        x = X
        out2 = forward_func(X,rand)

        try:
            r = r[:,:,0]
        except:
            pass

        if r.max()<0.1:
            region_mask = np.greater(r.flatten()+1.,0.5)
        else:
            region_mask = np.greater(r.flatten(),0.5)

        xx,xy = np.meshgrid(np.array(range(x.size(2))), np.array(range(x.size(3))) )
        xx = np.expand_dims(xx.flatten(),1)
        xy = np.expand_dims(xy.flatten(),1)
        xc = np.concatenate([xx,xy],1)
        
        xc = xc[region_mask,:]

        const2 = min(samps,xc.shape[0])


        global use_random
        if use_random:
            np.random.shuffle(xc)
        else:
            xc = xc[::(xc.shape[0]//const2),:]

        xx = xc[:const2,0]
        yy = xc[:const2,1]

        temp = X
        temp_list = [ temp[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
        temp = torch.cat(temp_list,2)

        l2 = []
        for i in range(len(out2)):

            temp = out2[i]

            if i>0 and out2[i].size(2) < out2[i-1].size(2):
                xx = xx/2.0
                yy = yy/2.0

            xx = np.clip(xx,0,temp.size(2)-1).astype(np.int32)
            yy = np.clip(yy,0,temp.size(3)-1).astype(np.int32)

            temp_list = [ temp[:,:, xx[j], yy[j]].unsqueeze(2).unsqueeze(3) for j in range(const2)]
            temp = torch.cat(temp_list,2)

            l2.append(temp.clone().detach())

        out2 = [torch.cat([li.contiguous() for li in l2],1)]

        return out2

    def forward_diff(self, X, inds=[1,3,5,8,11], rand=True):


        inds = self.inds
        l2 = self.forward_base(X,inds,rand)

        out2 = [l2[i].contiguous() for i in inds]


        for i in range(len(out2)):
            temp = out2[i]
            temp2 = F.pad(temp,(2,2,0,0),value=1.)
            temp3 = F.pad(temp,(0,0,2,2),value=1.)
            out2[i] = torch.cat([temp,temp2[:,:,:,4:],temp2[:,:,:,:-4],temp3[:,:,4:,:],temp3[:,:,:-4,:]],1)


        return out2