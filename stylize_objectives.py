import math

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from scipy.misc import imresize

from contextual_loss import *
import utils


class objective_class():

    def __init__(self, objective='remd_dp'):

        self.z_dist = torch.zeros(1).cuda()

        self.rand_ixx = {}
        self.rand_ixy = {}
        self.rand_iy = {}

        if objective == 'remd_dp':
            self.eval = self.gen_remd_dp_objective

        if objective == 'remd_dp_g':
            self.eval = self.gen_remd_dp_objective_guided

    def gen_remd_dp_objective_guided(self, z_x, z_c, z_s, gz, d_ell, d_weight=10.0, content_weight=4.0, moment_weight=1.0, style_loss_func=remd_loss, content_loss_func=dp_loss, h=1.0):


        ## Extract Random Subset of Features from Stylized Image & Content Image ##
        #    (Choose Features from Same locations in Stylized Image & Content Image) #
        final_loss = 0.
        #print('============')
        for ri in range(len(self.rand_ixx.keys())):
            xx, xy, yx = self.get_feature_inds(ri=ri)
            x_st, c_st = self.spatial_feature_extract(z_x, z_c, xx, xy)

            if gz.sum() > 0.:
                gxx, gxy = self.get_feature_inds_g()
                gx_st, gc_st = self.spatial_feature_extract(z_x, z_c, gxx, gxy)

            ## Reshape Features from Style Distribution ##
            d = z_s[ri][0].size(1)
            z_st = z_s[ri][0].view(1,d,-1,1)


            ## Compute Content Loss ##

            fm = 3+2*64+128*2+256*3+512*2

            if 1:
                ell_content = content_loss_func(x_st[:,:,:,:], c_st[:,:,:,:])
            else:
                ell_content = 0.5*torch.mean(torch.pow(x_st[:,:,:,:]-c_st[:,:,:,:],2))

            if 1:

                ## Compute Style Loss ##
                if 1:
                    if 1:
                        remd_loss, used_style_feats = style_loss_func(x_st[:,:fm,:,:], z_st[:,:fm,:,:], self.z_dist, splits=[fm])
                    else:
                        f1 = x_st[0,:fm,:,0]
                        f2 = z_st[0,:fm,:,0]
                        A = torch.mm(f1,f1.transpose(0,1))
                        G = torch.mm(f2,f2.transpose(0,1))
                        #wreck()
                        remd_loss = torch.sum(torch.pow(A-G,2))/(4*(f1.size(1)**2)*(f2.size(1)**2))
                        #style_weight = 1.

                else:
                    remd_loss = 0

                if gz.sum() > 0.:
                    for j in range(gz.size(2)):
                        remd_loss += style_loss_func(gx_st[:,:-2,j:(j+1),:], gz[:,:,j:(j+1),:],self.z_dist[:1]*0.)[0]/gz.size(2)

                ### Compute Moment Loss (constrains magnitude of features ###
                moment_ell = 0.
                if 1:
                    if gz.sum() > 0.:
                        moment_ell = moment_loss(torch.cat([x_st,gx_st],2)[:,:-2,:,:],torch.cat([z_st,gz],2),moments=[1,2])
                    else:
                        moment_ell = moment_loss(x_st[:,:-2,:,:],z_st,moments=[1,2])
                        
                #moment_ell = 0.
                content_weight_frac = 1./max(content_weight,1.)
                if 1:
                    moment_ell += content_weight_frac*style_loss_func(x_st[:,:3,:,:], z_st[:,:3,:,:],self.z_dist,splits=[3])[0]



                ell_style = remd_loss+moment_weight*moment_ell
                style_weight = 1.0 + moment_weight


            #remd_loss = 0
            ## Combine for Full Objective ##
            final_loss += (content_weight*ell_content+ell_style)/(content_weight+style_weight)
        
        return final_loss/len(self.rand_ixx.keys())


    def init_inds(self, z_x, z_s_all, r, ri):

        const = 128**2


        z_s = z_s_all[ri]


        try:
            temp = self.rand_ixx[ri]
        except:
            self.rand_ixx[ri]= []
            self.rand_ixy[ri]= []
            self.rand_iy[ri]= []

        #print(z_s[0].size())
        #print(len(z_s))
        for i in range(len(z_s)):

            d = z_s[i].size(1)
            z_st = z_s[i].view(1,d,-1,1)
            x_st = z_x[i]


            big_size = x_st.size(3)*x_st.size(2)



            stride_x = int(max(math.floor(math.sqrt(big_size//const)),1))
            offset_x = np.random.randint(stride_x)
            
            stride_y = int(max(math.ceil(math.sqrt(big_size//const)),1))
            offset_y = np.random.randint(stride_y)
            
            region_mask = r#.flatten()

            xx,xy = np.meshgrid(np.array(range(x_st.size(2)))[offset_x::stride_x], np.array(range(x_st.size(3)))[offset_y::stride_y] )
            
            xx = np.expand_dims(xx.flatten(),1)
            xy = np.expand_dims(xy.flatten(),1)
            xc = np.concatenate([xx,xy],1)

            try:
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]
            except:
                region_mask = region_mask[:,:,0]
                xc = xc[region_mask[xy[:,0],xx[:,0]],:]

            #np.random.shuffle(xc)

            self.rand_ixx[ri].append(xc[:,0])
            self.rand_ixy[ri].append(xc[:,1])

            zx = np.array(range(z_st.size(2))).astype(np.int32)
            #np.random.shuffle(zx)
            #print(z_st.size())

            self.rand_iy[ri].append(zx)

            #print('done')
            #wreck()

    def init_g_inds(self, coords, x_im):

        self.g_ixx = (coords[:,0]*x_im.size(2)).astype(np.int64)
        self.g_ixy = (coords[:,1]*x_im.size(3)).astype(np.int64)
   

    def spatial_feature_extract(self, z_x, z_c, xx, xy):

        l2 = []
        l3 = []

        for i in range(len(z_x)):

            temp = z_x[i]
            temp2 = z_c[i]

            if i>0 and z_x[i-1].size(2) > z_x[i].size(2):
                xx = xx/2.0#+0.5
                xy = xy/2.0#+0.5

            #xxt = np.clip(xx.astype(np.int32),0,temp.size(2)-1)
            #xyt = np.clip(xy.astype(np.int32),0,temp.size(3)-1)


            xxm = np.floor(xx).astype(np.float32)
            xxr = xx - xxm

            xym = np.floor(xy).astype(np.float32)
            xyr = xy - xym

            w00 = utils.to_device(torch.from_numpy((1.-xxr)*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w01 = utils.to_device(torch.from_numpy((1.-xxr)*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w10 = utils.to_device(torch.from_numpy(xxr*(1.-xyr))).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)
            w11 = utils.to_device(torch.from_numpy(xxr*xyr)).float().unsqueeze(0).unsqueeze(1).unsqueeze(3)


            xxm = np.clip(xxm.astype(np.int32),0,temp.size(2)-1)
            xym = np.clip(xym.astype(np.int32),0,temp.size(3)-1)

            s00 = xxm*temp.size(3)+xym
            s01 = xxm*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)
            s10 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+(xym)
            s11 = np.clip(xxm+1,0,temp.size(2)-1)*temp.size(3)+np.clip(xym+1,0,temp.size(3)-1)


            temp = temp.view(1,temp.size(1),temp.size(2)*temp.size(3),1)
            temp = temp[:,:,s00,:].mul_(w00).add_(temp[:,:,s01,:].mul_(w01)).add_(temp[:,:,s10,:].mul_(w10)).add_(temp[:,:,s11,:].mul_(w11))
            

            temp2 = temp2.view(1,temp2.size(1),temp2.size(2)*temp2.size(3),1)
            temp2 = temp2[:,:,s00,:].mul_(w00).add_(temp2[:,:,s01,:].mul_(w01)).add_(temp2[:,:,s10,:].mul_(w10)).add_(temp2[:,:,s11,:].mul_(w11))
            #wreck()

            l2.append(temp)
            l3.append(temp2)

        x_st = torch.cat([li.contiguous() for li in l2],1)
        c_st = torch.cat([li.contiguous() for li in l3],1)


        xx = torch.from_numpy(xx).cuda().view(1,1,x_st.size(2),1).float()
        yy = torch.from_numpy(xy).cuda().view(1,1,x_st.size(2),1).float()


        x_st = torch.cat([x_st,xx,yy],1)
        c_st = torch.cat([c_st,xx,yy],1)

        return x_st, c_st




    def shuffle_feature_inds(self, i=0):

        for ri in self.rand_ixx.keys():
            np.random.shuffle(self.rand_ixx[ri][i])
            np.random.shuffle(self.rand_ixy[ri][i])
            np.random.shuffle(self.rand_iy[ri][i])


    def get_feature_inds(self,ri=0, i=0, cnt=32**2):

        xx = self.rand_ixx[ri][i][:cnt]
        xy = self.rand_ixy[ri][i][:cnt]
        yx = self.rand_iy[ri][i][:cnt]

        return xx, xy, yx
    
    def get_feature_inds_g(self, i=0, cnt=32**2):

        xx = self.g_ixx
        xy = self.g_ixy

        return xx, xy
