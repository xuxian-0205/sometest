import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from core.utils.utils import bilinear_sampler, coords_grid,bilinear_samplere

try:
    import alt_cuda_corr
except:
    # alt_cuda_corr is not compiled
    pass
import torch.nn as nn

class CorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)
        
        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i].float()
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1*wd1)
        fmap2 = fmap2.view(batch, dim, ht2*wd2)
        
        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr  / torch.sqrt(torch.tensor(dim).float())




class AlternateCorrBlock:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius

        self.pyramid = [(fmap1, fmap2)]
        for i in range(self.num_levels):
            fmap1 = F.avg_pool2d(fmap1, 2, stride=2)
            fmap2 = F.avg_pool2d(fmap2, 2, stride=2)
            self.pyramid.append((fmap1, fmap2))

    def __call__(self, coords):
        coords = coords.permute(0, 2, 3, 1)
        B, H, W, _ = coords.shape
        dim = self.pyramid[0][0].shape[1]

        corr_list = []
        for i in range(self.num_levels):
            r = self.radius
            fmap1_i = self.pyramid[0][0].permute(0, 2, 3, 1).contiguous()
            fmap2_i = self.pyramid[i][1].permute(0, 2, 3, 1).contiguous()

            coords_i = (coords / 2**i).reshape(B, 1, H, W, 2).contiguous()
            corr, = alt_cuda_corr.forward(fmap1_i, fmap2_i, coords_i, r)
            corr_list.append(corr.squeeze(1))

        corr = torch.stack(corr_list, dim=1)
        corr = corr.reshape(B, -1, H, W)
        return corr / torch.sqrt(torch.tensor(dim).float())




class CorrpyBlockTTC:
    def __init__(self, fmap1, fmap2,net,inp, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.Fmap1 = fmap1
        self.Fmap2 = fmap2
        self.net = net
        self.inp = inp
        self.rate = [0.5, 0.75, 1, 1.25] # 四层

    def __call__(self, corrd0,corrd1,bl,ba,boxnum, exp):
        
        bacorr_all = []
        bacorr_notf_all = []
        corrnet_all = []
        corrinp_all = []
        for iba in range(ba):  #ba = batch
            if boxnum[iba]>0:
                feature1 = []
                feather0 = self.Fmap1[0][iba].unsqueeze(0)

                feature1.append(self.Fmap2[0][iba].unsqueeze(0))
                feature1.append(self.Fmap2[1][iba].unsqueeze(0))
                feature1.append(self.Fmap2[2][iba].unsqueeze(0))
                feature1.append(self.Fmap2[3][iba].unsqueeze(0))

                feathernet = self.net[iba].unsqueeze(0)
                featherinp = self.inp[iba].unsqueeze(0)

                coordsu0 = torch.cat([corrd0[i,:,:,:].unsqueeze(0) for i in range(boxnum[iba])])
                coordsu0 = coordsu0.permute(0,2,3,1)
                coordsu0 = coordsu0.contiguous().view(1, boxnum[iba]*10*10,1,2)

                coordsu1 = torch.cat([corrd1[i,:,:,:].unsqueeze(0) for i in range(boxnum[iba])])
                coordsu1 = coordsu1.permute(0,2,3,1)
                coordsu1 = coordsu1.contiguous().view(boxnum[iba]*10*10,1,1,2)

                

                corrnet = bilinear_sampler(feathernet, coordsu0).contiguous().view(192,boxnum[iba],10,10).permute(1,0,2,3)
                corrinp = bilinear_sampler(featherinp, coordsu1.permute(1,0,2,3)).contiguous().view(192,boxnum[iba],10,10).permute(1,0,2,3)
                corrnet_all.append(corrnet)
                corrinp_all.append(corrinp)

               
                expsu = torch.cat([exp[i,:,:,:].unsqueeze(0) for i in range(boxnum[iba])])
                expsu = expsu.permute(0, 2, 3, 1)
                expsu = expsu.contiguous().view(boxnum[iba] * 10 * 10, 1, 1, 1)

                h1, w1 = 10, 10

                

                r2 = 1
                de = torch.linspace(-r2, r2, 2 * r2 + 1)  # -1 1 3
                de1 = torch.zeros(1)
                centrexp_lvl = expsu.reshape(boxnum[iba]*h1 * w1, 1, 1, 1).to(coordsu0.device)
                delte = torch.stack(torch.meshgrid(de, de1), axis=-1).to(coordsu0.device)
                delte_lvl = delte.view(1, 2 * r2 + 1, 1, 2)
                exp_lvl = delte_lvl + centrexp_lvl
                exp_lvl[:, :, :, 1] = 0


               
                corr = bilinear_sampler(feather0, coordsu0)
                corr = corr.permute(2,3,0,1).view(boxnum[iba]*10*10,1,1,256)
                corr = corr.permute(0,3,1,2)  #N,C,W,H

                
                r3 = 5
                
                corr_feature = []
                for i in [0, 1, 2, 3]:

                   

                    dx = torch.linspace(-r3, r3, 2 * r3 + 1)*self.rate[i]
                    dy = torch.linspace(-r3, r3, 2 * r3 + 1)*self.rate[i]
                    delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coordsu1.device)
                    delta_lvl = delta.view(1, 2 * r3 + 1, 2 * r3 + 1, 2)

                    coords_lvl = coordsu1*self.rate[i] + delta_lvl
                    coords_lvl = coords_lvl.contiguous().view(1, boxnum[iba] * 10 * 10, (2 * r3 + 1) * (2 * r3 + 1), 2)

                    corr2 = bilinear_sampler(feature1[i], coords_lvl)
                    corr2 = corr2.permute(2, 3, 0, 1).contiguous().view(boxnum[iba] * 10 * 10, 2 * r3 + 1, 2 * r3 + 1, 256)
                    corr2 = corr2.permute(0, 3, 1, 2)
                  
                    correlation = self.corr(corr, corr2)  
                    corr_feature.append(correlation.view(boxnum[iba], 10, 10,1, (2 * r3 + 1)*(2 * r3 + 1)))
                corr_featureall = torch.cat([cor for cor in corr_feature],dim=3)
                bacorr_notf_all.append(corr_feature[2].squeeze(3))
                
                corr_featureall = corr_featureall.view(boxnum[iba] * h1 * w1, 4, (2 * r3 + 1) * (2 * r3 + 1)).permute(0,2, 1).unsqueeze(3)
                out = bilinear_samplere(corr_featureall, exp_lvl, mode='bilinear').view(boxnum[iba], h1, w1, (2 * r3 + 1) * (2 * r3 + 1) * (2 * r2 + 1))
                bacorr_all.append(out)

        bacorr_notf_all = torch.cat([cor for cor in bacorr_notf_all], dim = 0).permute(0,3,1,2)
        bacorr_all = torch.cat([cor for cor in bacorr_all], dim=0).permute(0,3,1,2)
        bacorr = torch.cat([bacorr_notf_all,bacorr_all],dim=1)

        corrnet_all = torch.cat([cor for cor in corrnet_all], dim=0)
        corrinp_all = torch.cat([cor for cor in corrinp_all], dim=0)
        return bacorr,corrnet_all,corrinp_all

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht1, wd1 = fmap1.shape
        batch, dim, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch, dim, ht1 * wd1)
        fmap2 = fmap2.view(batch, dim, ht2 * wd2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht1, wd1, 1, ht2, wd2)
        return corr / torch.sqrt(torch.tensor(dim).float())

class CorrpyBlock4_3_343:
    def __init__(self, fmap1, fmap2, num_levels=4, radius=4):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.corr_pyramid2 = []
        self.map1wh = []
        for dix in [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)]:
            # all pairs correlation
            corr = CorrBlock.corr(fmap1[int(dix[0])], fmap2[int(dix[1])])
            # F.upsample(dchange2, [im.size()[2], im.size()[3]], mode='bilinear')
            
            batch, h1, w1, dim, h2, w2 = corr.shape
            self.map1wh.append([h1, w1])
            corr = corr.reshape(batch * h1 * w1, dim, h2, w2)
            
            self.corr_pyramid.append(corr)
        corr2 = self.corr_pyramid[2]
        for i in range(self.num_levels - 1):
            corr2 = F.avg_pool2d(corr2, 2, stride=2)
            self.corr_pyramid2.append(corr2)
        self.rate = [0.5, 0.75, 1, 1.25, 1.5]

    def __call__(self, coords, exp):
        r = 3
        r2 = 1
        r3 = 3
        coords = coords.permute(0, 2, 3, 1) 
        exp = exp.squeeze(1)
        exp = exp.unsqueeze(3)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        out_pyramid2 = []
        
        de = torch.linspace(-r2, r2, 2 * r2 + 1)
        de1 = torch.zeros(1)
        centrexp_lvl = exp.reshape(batch * h1 * w1, 1, 1, 1)
        delte = torch.stack(torch.meshgrid(de, de1), axis=-1).to(coords.device)
        delte_lvl = delte.view(1, 2 * r2 + 1, 1, 2)
        exp_lvl = delte_lvl + centrexp_lvl
        exp_lvl[:, :, :, 1] = 0

        
        for i in [0, 1, 2, 3, 4]:
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r3, r3, 2 * r3 + 1) * self.rate[i]
            dy = torch.linspace(-r3, r3, 2 * r3 + 1) * self.rate[i]
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) * self.rate[i]

            delta_lvl = delta.view(1, 2 * r3 + 1, 2 * r3 + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl
            corr = bilinear_sampler(corr, coords_lvl)
            out_pyramid.append(corr)
        pyramid = torch.cat(out_pyramid, dim=1)
        pyramid = pyramid.view(batch * h1 * w1, 5, (2 * r3 + 1) * (2 * r3 + 1)).permute(0, 2, 1).unsqueeze(3)
        out = bilinear_samplere(pyramid, exp_lvl, mode='bilinear').view(batch, h1, w1,
                                                                        (2 * r3 + 1) * (2 * r3 + 1) * (2 * r2 + 1))

        '''
        import cv2
        pyramid6 = pyramid.view(batch,h1 , w1, (2 * r3 + 1)*(2 * r3 + 1),5)
        imgshow = pyramid6.detach().cpu().numpy()
        imshow24 = imgshow[0, :, :, 24, :]*0.071+imgshow[0, :, :, 23, :]*0.056+imgshow[0, :, :, 25, :]*0.056+imgshow[0, :, :, 16, :]*0.045+imgshow[0, :, :, 17, :]*0.056+imgshow[0, :, :, 18, :]*0.045+imgshow[0, :, :, 30, :]*0.045+imgshow[0, :, :, 31, :]*0.056+imgshow[0, :, :, 32, :]*0.045
        imgshow0 =imshow24[:,:,0]
        imgshow1 =imshow24[:,:,1]
        imgshow2 =imshow24[:,:,2]
        imgshow3 =imshow24[:,:,3]
        imgshow4 =imshow24[:,:,4]


        im0 = 255*(imgshow0-imshow24.min())/imshow24.max()
        im1 = 255*(imgshow1 - imshow24.min()) / imshow24.max()
        im2 = 255*(imgshow2 - imshow24.min()) / imshow24.max()
        im3 = 255*(imgshow3 - imshow24.min()) / imshow24.max()
        im4 = 255*(imgshow4 - imshow24.min()) / imshow24.max()


        k = 0.95
        plt.imshow(imgshow0, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        #plt.colorbar()
        plt.savefig('/home/pic_show/0.png')
        plt.show()

        plt.imshow(imgshow1, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/pic_show/1.png')
        plt.show()
        plt.imshow(imgshow2, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/pic_show/2.png')
        plt.show()
        plt.imshow(imgshow3, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/pic_show/3.png')
        plt.show()
        plt.imshow(imgshow4, vmax=imshow24.max()*k, cmap=plt.cm.jet)
        plt.savefig('/home/pic_show/4.png')
        plt.show()
        '''

        # 原尺度的
        corr = self.corr_pyramid[2]
        dx = torch.linspace(-r, r, 2 * r + 1)
        dy = torch.linspace(-r, r, 2 * r + 1)
        delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

        centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2)
        delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
        coords_lvl = centroid_lvl + delta_lvl

        corr = bilinear_sampler(corr, coords_lvl)
        corr = corr.view(batch, h1, w1, -1)
        out_pyramid2.append(corr)
        for i in range(self.num_levels - 1):
            corr = self.corr_pyramid2[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** (i + 1)
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid2.append(corr)
        out2 = torch.cat(out_pyramid2, dim=-1)
        out3 = torch.cat([out2, out], dim=-1)

        return out3.permute(0, 3, 1, 2).contiguous().float()

    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd)
        fmap2 = fmap2.view(batch, dim, ht * wd)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())