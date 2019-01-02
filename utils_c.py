import time
import shutil
import os

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from scipy.misc import imread,imsave


def match_device(ref, mut):
    if ref.is_cuda and not mut.is_cuda:
        mut = mut.cuda()

    if not ref.is_cuda and mut.is_cuda:
        mut = mut.cpu()

    return mut

#Define YUV color transform
C = torch.from_numpy(np.float32([[0.577350,0.577350,0.577350],[-0.577350,0.788675,-0.211325],[-0.577350,-0.211325,0.788675]]))

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

def save_checkpoint(state, is_best, filename='./nn_models/checkpoint.pth.tar'):
    torch.save(state, filename)
  
def load_checkpoint(filename, model, optimizer):
    
    steps = 0
    best_loss = 1e10

    if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            steps = checkpoint['steps']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (# steps {})".format(filename,steps))

    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return best_loss,steps

def load_checkpoint_nn_only(filename, model):
    if os.path.isfile(filename):
            print("=> loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' ".format(filename))

    else:
        print("=> no checkpoint found at '{}'".format(filename))