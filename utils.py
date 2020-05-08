import time
import shutil
import os
import sys

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from imageio import imread
use_gpu = True


def match_device(ref, mut):
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

#Define YUV color transform
C = torch.from_numpy(np.float32([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]))

def aug_canvas(canvas, scale, s_iter):

    h = canvas.size(2)
    w = canvas.size(3)

    mx = max(h,w)

    frac = 512./float(mx)

    h = int(h*frac)
    w = int(w*frac)

    prg = 0.
    if scale < 4:
        prg = (scale-1)*0.21+s_iter/250.*0.21
    else:
        prg = 0.63+s_iter/250.*0.37
    prg = int(prg*100.)

    canvas = F.upsample(canvas,(h,w),mode='bilinear').cpu()

    canvas = torch.clamp(canvas[0],0.,1.).data.numpy().transpose(1,2,0)

    return np.uint8(canvas*255)

def rgb_to_yuv(rgb):

    global C
    C = match_device(rgb,C)
    C_t = torch.t(C)

    rgb_rs = rgb.view(-1,3)
    yuv_rs  = torch.mm(rgb_rs,C_t)
    yuv = yuv_rs.view(rgb.shape[0],rgb.shape[1],3)

    return yuv

def rgb_to_yuv_pc(rgb):

    global C
    C = match_device(rgb,C)

    yuv  = torch.mm(C,rgb)

    return yuv

def yuv_to_rgb(rgb):

    global C
    C = match_device(rgb,C)

    rgb_rs = rgb.view(-1,3)
    yuv_rs  = torch.mm(rgb_rs,C)
    yuv = yuv_rs.view(rgb.shape[0],rgb.shape[1],3)

    return yuv


def to_device(tensor):
    if use_gpu:
        return tensor.cuda()
    else:
        return tensor

def match_device(ref, mut):
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

def split_99(x,y):

    w = x.size(2)
    h = x.size(3)

    r0 = 0
    r1 = w//4
    r2 = w//2
    rs = [r0,r1,r2]

    c0 = 0
    c1 = h//4
    c2 = h//2
    cs = [c0,c1,c2]

    wt = w//2
    ht = h//2

    xo = []
    yo = []

    for i in range(3):
        for j in range(3):
            sr = rs[i]
            sc = cs[j]

            er = sr + ht
            ec = sc + wt

            xo.append(x[:,:,sr:er,sc:ec])
            yo.append(y[:,:,sr:er,sc:ec])

    return xo,yo

def build_guidance(content_path,style_path,coords,augment=True):
    
    im_a = imread(content_path)
    im_b = imread(style_path)

    coords_true = coords.copy()
    coords_true[:,0] = coords[:,1]
    coords_true[:,1] = coords[:,0]
    coords_true[:,2] = coords[:,3]
    coords_true[:,3] = coords[:,2]
    coords = coords_true

    if 1:
        new_coords = []

        rng=1
        if not augment:
            rng = 0
            
        dilation = 0.01
        for c in range(coords.shape[0]):
            oc = coords[c:c+1,:]
            for i in range(rng*2+1):
                for j in range(rng*2+1):
                    nc = np.zeros((1,4))

                    ma = max(im_a.shape[0],im_a.shape[1])
                    mb = max(im_b.shape[0],im_b.shape[1])

                    nc[0,0] = oc[0,0]+(-rng+i)*int(dilation*ma)
                    nc[0,1] = oc[0,1]+(-rng+j)*int(dilation*ma)
                    nc[0,2] = oc[0,2]+(-rng+i)*int(dilation*mb)
                    nc[0,3] = oc[0,3]+(-rng+j)*int(dilation*mb)

                    nc[0,0] = np.clip(nc[0,0],0,im_a.shape[0] )
                    nc[0,1] = np.clip(nc[0,1],0,im_a.shape[1])
                    nc[0,2] = np.clip(nc[0,2],0,im_b.shape[0] )
                    nc[0,3] = np.clip(nc[0,3],0,im_b.shape[1])

                    new_coords.append(nc)

        coords = np.concatenate(new_coords,0)

    coords[:,0] = coords[:,0]/im_a.shape[0] 
    coords[:,1] = coords[:,1]/im_a.shape[1] 
    coords[:,2] = coords[:,2]/im_b.shape[0] 
    coords[:,3] = coords[:,3]/im_b.shape[1] 

    return coords

def extract_regions(content_path,style_path):
    s_regions = imread(style_path).transpose(1,0,2)
    c_regions = imread(content_path).transpose(1,0,2)

    color_codes,c1 = np.unique(s_regions.reshape(-1, s_regions.shape[2]), axis=0,return_counts=True)

    color_codes = color_codes[c1>10000]

    c_out = []
    s_out = []

    for c in color_codes:
        c_expand =  np.expand_dims(np.expand_dims(c,0),0)
        
        s_mask = np.equal(np.sum(s_regions - c_expand,axis=2),0).astype(np.float32)
        c_mask = np.equal(np.sum(c_regions - c_expand,axis=2),0).astype(np.float32)

        s_out.append(s_mask)
        c_out.append(c_mask)

    return [c_out,s_out]


big_patch_sz=256

def load_path_for_pytorch(path, max_side=1000, force_scale=False, verbose=True):

    com_f = max

    x = imread(path)
    s = x.shape

    x = x/255.#-0.5
    xt = x.copy()
    
    if len(s) < 3:
        x = np.stack([x,x,x],2)

    if x.shape[2] > 3:
        x = x[:,:,:3]

    x = x.astype(np.float32)
    x = torch.from_numpy(x).contiguous().permute(2,0,1).contiguous()


    if (com_f(s[:2])>max_side and max_side>0) or force_scale:


        fac = float(max_side)/com_f(s[:2])
        x = F.upsample(x.unsqueeze(0),( int(s[0]*fac), int(s[1]*fac) ), mode='bilinear')[0]
        so = s
        s = x.shape
        #if verbose:
        #    print(so)
        #    print(s)
        #    print('-----')


    return x


def load_style_guidance(phi,path,coords_t,scale):


    style_im = to_device(Variable(load_path_for_pytorch(path, max_side=scale, verbose=False, force_scale=True).unsqueeze(0)))

    coords = coords_t.copy()
    coords[:,0]=coords[:,0]*style_im.size(2)
    coords[:,1]=coords[:,1]*style_im.size(3)
    coords = coords.astype(np.int64)

    xx = coords[:,0]
    xy = coords[:,1]

    zt = phi(style_im)
    
    l2 = []

    for i in range(len(zt)):

        temp = zt[i]

        if i>0 and zt[i-1].size(2) > zt[i].size(2):
            xx = xx/2.0
            xy = xy/2.0

        xxm = np.floor(xx).astype(np.float32)
        xxr = xx - xxm

        xym = np.floor(xy).astype(np.float32)
        xyr = xy - xym

        w00 = to_device(torch.from_numpy((1.-xxr)*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
        w01 = to_device(torch.from_numpy((1.-xxr)*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
        w10 = to_device(torch.from_numpy(xxr*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
        w11 = to_device(torch.from_numpy(xxr*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)


        xxm = np.clip(xxm.astype(np.int32),0,temp.size(2)-1)
        xym = np.clip(xym.astype(np.int32),0,temp.size(3)-1)

        s00 = xxm*temp.size(3)+xym
        s01 = xxm*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)
        s10 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+(xym)
        s11 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)


        temp = temp.view(1,temp.size(1),temp.size(2)*temp.size(3),1)
        temp = temp[:,:,s00,:].mul_(w00).add_(temp[:,:,s01,:].mul_(w01)).add_(temp[:,:,s10,:].mul_(w10)).add_(temp[:,:,s11,:].mul_(w11))
        
        l2.append(temp)
    gz = torch.cat([li.contiguous() for li in l2],1)

    return gz

def load_style_folder(phi, paths, regions, ri, n_samps=-1,subsamps=-1,scale=-1, inner=1, cpu_mode=False):

    if n_samps > 0:
        list.sort(paths)
        paths = paths[::max((len(paths)//n_samps),1)]
    else:
        pass#print(len(paths))
        
    total_sum = 0.
    z = []
    z_ims = []
    nloaded = 0
    for p in paths:

        nloaded += 1
        
        style_im = to_device(Variable(load_path_for_pytorch(p, max_side=scale, verbose=False, force_scale=True).unsqueeze(0), requires_grad=False))
        
        r_temp = regions[1][ri]
        if len(r_temp.shape) > 2:
            r_temp = r_temp[:,:,0]

        r_temp = torch.from_numpy(r_temp).unsqueeze(0).unsqueeze(0).contiguous()
        #print(r_temp.size())
        r = F.upsample(r_temp,(style_im.size(3),style_im.size(2)),mode='bilinear')[0,0,:,:].numpy()        
        sts = [style_im]

        z_ims.append(style_im)

        for j in range(inner):

            style_im = sts[np.random.randint(0,len(sts))]
            
            with torch.no_grad():
                zt = phi(style_im,subsamps,r)
                
            zt = [li.view(li.size(0),li.size(1),-1,1) for li in zt]

            if len(z) == 0:
                z = zt

            else:
                z = [torch.cat([zt[i],z[i]],2) for i in range(len(z))]

    return z, z_ims
