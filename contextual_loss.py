import torch
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
import utils

def pairwise_distances_sq_l2(x, y):

    x_norm = (x**2).sum(1).view(-1, 1)
    y_t = torch.transpose(y, 0, 1)
    y_norm = (y**2).sum(1).view(1, -1)
    
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)

    return torch.clamp(dist, 1e-5, 1e5)/x.size(1)


def pairwise_distances_cos(x, y):

    x_norm = torch.sqrt((x**2).sum(1).view(-1, 1))
    y_t = torch.transpose(y, 0, 1)
    y_norm = torch.sqrt((y**2).sum(1).view(1, -1))
    
    dist = 1.-torch.mm(x, y_t)/x_norm/y_norm

    return dist

def get_DMat(X,Y,h=1.0,cb=0,splits=[128*3+256*3+512*4], cos_d=True):
    n = X.size(0)
    m = Y.size(0)
    M = utils.to_device(Variable(torch.zeros(n,m)))


    if 1:
        cb = 0
        ce = 0
        for i in range(len(splits)):
            if cos_d:
                ce = cb + splits[i]
                M = M + pairwise_distances_cos(X[:,cb:ce],Y[:,cb:ce])
            
                cb = ce
            else:
                ce = cb + splits[i]
                M = M + torch.sqrt(pairwise_distances_sq_l2(X[:,cb:ce],Y[:,cb:ce]))
            
                cb = ce

    return M


def viz_d(zx,coords):


    viz = zx[0][:,:1,:,:].clone()*0.

    for i in range(coords.shape[0]):
        vizt = zx[0][:,:1,:,:].clone()*0.

        for z in zx:
            cx = int(coords[i,0]*z.size(2))
            cy = int(coords[i,1]*z.size(3))

            anch = z[:,:,cx:cx+1,cy:cy+1]
            x_norm = torch.sqrt((z**2).sum(1,keepdim=True))
            y_norm = torch.sqrt((anch**2).sum(1,keepdim=True))
            dz = torch.sum(z*anch,1,keepdim=True)/x_norm/y_norm
            vizt = vizt+F.upsample(dz,(viz.size(2),viz.size(3)),mode='bilinear')*z.size(1)

        viz = torch.max(viz,vizt/torch.max(vizt))

    vis_o = viz.clone()
    viz = viz.data.cpu().numpy()[0,0,:,:]/len(zx)
    return vis_o

def remd_loss(X,Y, h=None, cos_d=True, splits= [3+64+64+128+128+256+256+256+512+512],return_mat=False):

    d = X.size(1)


    if d == 3:
        X = utils.rgb_to_yuv_pc(X.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        Y = utils.rgb_to_yuv_pc(Y.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)

    else:
        X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    #Relaxed EMD
    CX_M = get_DMat(X,Y,1.,cos_d=cos_d, splits=splits)
    
    if return_mat:
        return CX_M
    
    if d==3:
        CX_M = CX_M+get_DMat(X,Y,1.,cos_d=False, splits=splits)

    m1,m1_inds = CX_M.min(1)
    m2,m2_inds = CX_M.min(0)

    if m1.mean() > m2.mean():
        used_style_feats = Y[m1_inds,:]
    else:
        used_style_feats = Y

    remd = torch.max(m1.mean(),m2.mean())

    return remd, used_style_feats


def remd_loss_g(X,Y, GX, GY, h=1.0, splits= [3+64+64+128+128+256+256+256+512+512]):

    d = X.size(1)

    if d == 3:
        X = utils.rgb_to_yuv_pc(X.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        Y = utils.rgb_to_yuv_pc(Y.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        GX = utils.rgb_to_yuv_pc(GX.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)
        GY = utils.rgb_to_yuv_pc(GY.transpose(0,1).contiguous().view(d,-1)).transpose(0,1)


    else:
        X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        GX = GX.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
        GY = GY.transpose(0,1).contiguous().view(d,-1).transpose(0,1)


    c1 = 10000.
    c2 = 1.
    
    CX_M = get_DMat(X,Y,1.,cos_d=True, splits=splits)

    if d==3:
        CX_M = CX_M+get_DMat(X,Y,1.,cos_d=False, splits=splits)


    CX_M_2 = get_DMat(GX,GY,1.,cos_d=True, splits=splits)+get_DMat(GX,GY,1.,cos_d=False, splits=splits)#CX_M[i:,i:].clone()
    for i in range(GX.size(0)-1):
        CX_M_2[(i+1):,i] = CX_M_2[(i+1):,i]*1000.
        CX_M_2[i,(i+1):] = CX_M_2[i,(i+1):]*1000.


    m1,m1_inds = CX_M.min(1)
    m2,m2_inds = CX_M.min(0)
    m2,min_inds = torch.topk(m2,m1.size(0),largest=False)

    if m1.mean() > m2.mean():
        used_style_feats = Y[m1_inds,:]
    else:
        used_style_feats = Y[min_inds,:]

    m12,_ = CX_M_2.min(1)
    m22,_ = CX_M_2.min(0)

    used_style_feats = Y[m1_inds,:]
    remd = torch.max(m1.mean()*h,m2.mean())+c2*torch.max(m12.mean()*h,m22.mean())

    return remd, used_style_feats


def moment_loss(X,Y,moments=[1,2]):

    d = X.size(1)
    ell = 0.

    Xo = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Yo = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    splits = [Xo.size(1)]

    cb = 0
    ce = 0
    for i in range(len(splits)):
        ce = cb + splits[i]
        X = Xo[:,cb:ce]
        Y = Yo[:,cb:ce]
        cb = ce

        mu_x = torch.mean(X,0,keepdim=True)
        mu_y = torch.mean(Y,0,keepdim=True)
        mu_d = torch.abs(mu_x-mu_y).mean()



        if 1 in moments:
            ell = ell + mu_d


        if 2 in moments:
            sig_x = torch.mm((X-mu_x).transpose(0,1), (X-mu_x))/X.size(0)
            sig_y = torch.mm((Y-mu_y).transpose(0,1), (Y-mu_y))/Y.size(0)


            sig_d = torch.abs(sig_x-sig_y).mean()
            ell = ell + sig_d


    return ell


def moment_loss_g(X,Y,GX,moments=[1,2]):

    d = X.size(1)
    ell = 0.

    Xo = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Yo = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    GXo = GX.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    betas = torch.pow(get_DMat(Xo, GXo),1)
    betas,_ = torch.max(betas,1)
    betas = betas.unsqueeze(1).detach()
    betas = betas*torch.ge(betas,0.2).float()

    splits = [Xo.size(1)]
    cb = 0
    ce = 0
    for i in range(len(splits)):
        ce = cb + splits[i]
        X = Xo[:,cb:ce]
        Y = Yo[:,cb:ce]
        cb = ce

        mu_x = torch.sum(betas*X,0,keepdim=True)/torch.sum(betas)
        mu_y = torch.mean(Y,0,keepdim=True)
        mu_d = torch.abs(mu_x-mu_y).mean()



        if 1 in moments:
            ell = ell + mu_d


        if 2 in moments:
            sig_x = torch.mm(((betas*X-mu_x)).transpose(0,1), (betas*X-mu_x))/torch.sum(torch.pow(betas,2))
            sig_y = torch.mm((Y-mu_y).transpose(0,1), (Y-mu_y))/Y.size(0)


            sig_d = torch.abs(sig_x-sig_y).mean()
            ell = ell + sig_d

    return ell

def dp_loss(X,Y):

    d = X.size(1)

    X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Xc = X[:,-2:]
    Y = Y[:,:-2]
    X = X[:,:-2]

    if 0:
        dM = torch.exp(-2.*get_DMat(Xc,Xc,1., cos_d=False))
        dM = dM/dM.sum(0,keepdim=True).detach()*dM.size(0)
    else:
        dM = 1.

    Mx = get_DMat(X,X,1.,cos_d=True,splits=[X.size(1)])
    Mx = Mx/Mx.sum(0,keepdim=True)

    My = get_DMat(Y,Y,1.,cos_d=True,splits=[X.size(1)])
    My = My/My.sum(0,keepdim=True)

    d = torch.abs(dM*(Mx-My)).mean()*X.size(0)

    return d


def dp_loss_warp(X,Y):

    d = X.size(1)

    X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    Xc = X[:,-2:]
    Y = Y[:,:-2]
    X = X[:,:-2]

    if 0:
        dM = torch.exp(-2.*get_DMat(Xc,Xc,1., cos_d=False))
        dM = dM/dM.sum(0,keepdim=True).detach()*dM.size(0)
    else:
        dM = 1.

    Mx = get_DMat(X,X,1.,cos_d=False,splits=[X.size(1)])
    Mx = Mx/Mx.sum(0,keepdim=True)

    My = get_DMat(Y,Y,1.,cos_d=False,splits=[X.size(1)])
    My = My/My.sum(0,keepdim=True)

    d = torch.abs(dM*(Mx-My)).mean()*X.size(0)

    return d




def dp_loss_g(X,Y,GX):

    d = X.size(1)

    X = X.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    Y = Y.transpose(0,1).contiguous().view(d,-1).transpose(0,1)
    GX = GX.transpose(0,1).contiguous().view(d,-1).transpose(0,1)

    betas,_ = torch.max(torch.pow(get_DMat(X, GX),1),1)
    betas = betas.unsqueeze(1).detach()
    betas = torch.matmul(betas,betas.transpose(0,1))

    Mx = get_DMat(X,X,1.,splits=[X.size(1)])
    Mx = Mx/Mx.sum(0,keepdim=True)

    My = get_DMat(Y,Y,1.,splits=[X.size(1)])
    My = My/My.sum(0,keepdim=True)

    d = torch.abs(betas*(Mx-My)).sum(0).mean()


    return d
