import torch
from torch.autograd import Variable
import torch.nn.functional as F

def dec_lap_pyr(X,levs):
    pyr = []
    cur = X
    for i in range(levs):
        cur_x = cur.size(2)
        cur_y = cur.size(3)

        x_small = F.upsample_bilinear(cur, (max(cur_x//2,1), max(cur_y//2,1)))
        x_back  = F.upsample_bilinear(x_small, (cur_x,cur_y))
        lap = cur - x_back
        pyr.append(lap)
        cur = x_small

    pyr.append(cur)

    return pyr

def syn_lap_pyr(pyr):

    cur = pyr[-1]
    levs = len(pyr)
    for i in range(0,levs-1)[::-1]:
        up_x = pyr[i].size(2)
        up_y = pyr[i].size(3)
        cur = pyr[i] + F.upsample_bilinear(cur,(up_x,up_y))

    return cur