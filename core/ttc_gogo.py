import numpy as np
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from core.update import  SmallUpdateBlock, ScaleflowUpdateBlock, DCUpdateBlock
from core.extractor import BasicEncoder, SmallEncoder
from core.ttcnet import TTCEncoder,maskHead,midHead,flowHead,preHead
from core.corr import CorrpyBlockTTC
import cv2
from utils.utils import  coords_grid,bilinear_sampler
import GNN_model
import model_cls

# pt_orig = np.array(([[bl[b,l,2].cpu(),bl[b,l,4].cpu()], [bl[b,l,2].cpu(),bl[b,l,5].cpu()], [bl[b,l,3].cpu(),bl[b,l,5].cpu()]])).astype(np.float32)
# pt_dest = np.array(([[bl[b,l,6].cpu(),bl[b,l,8].cpu()], [bl[b,l,6].cpu(),bl[b,l,9].cpu()], [bl[b,l,7].cpu(),bl[b,l,9].cpu()]])).astype(np.float32)
# Transfor = cv2.getAffineTransform(pt_orig, pt_dest)
# Transform.append([b,Transfor])
# pt_ans = np.dot(Transfor[:,0:2],pt_orig.T)+Transfor[:,2:]

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass


# best_weight raft-kitti_11.pth
def gaussian2D2(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h/h.sum()

def exp2grid(exp):
    #exp = exp.clamp(0, 3)
    x = exp * 4 - 2
    return x

def grid2exp(exp):
    exp = exp.clamp(0, 3)
    x = exp * 0.25 + 0.5
    return x

class TTC(nn.Module):
    def __init__(self, args):
        super(TTC, self).__init__()
        self.args = args

        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3

        else:
            self.hidden_dim = hdim = 192
            self.context_dim = cdim = 192
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        if 'alternate_corr' not in self.args:
            self.args.alternate_corr = False

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)
            self.cnet = SmallEncoder(output_dim=hdim + cdim, norm_fn='none', dropout=args.dropout)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout)
            self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch', dropout=args.dropout)

        #self.maskhead_ = maskHead(input_dim=1024)

        self.encoder_corr = preHead(input_dim=484, out_dim=256)
        self.encoder_cont = preHead(input_dim=384, out_dim=128)
        self.encoder_mot = preHead(input_dim=3, out_dim=64)

        self.flowhead = flowHead(input_dim=514)
        self.midhead = midHead(input_dim=512)
        self.maskhead = maskHead(input_dim=706)

        # blur1 = cv2.resize(blur1, (int(w * (xita / 2)), int(h * (xita / 2))))
        xita = 2 ** 0.25  # 0.25 + 1
        self.delta1 = 1.75 / 2
        kernel1 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 0.5  # 0.5 + 1
        self.delta2 = 1.5 / 2
        kernel2 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 0.75  # 0.75 + 1
        self.delta3 = 1.25 / 2
        kernel3 = gaussian2D2([5, 5], sigma=(xita, xita))
        xita = 2 ** 1  # 1 + 1
        self.delta4 = 1 / 2
        kernel4 = gaussian2D2([5, 5], sigma=(xita, xita))

        kernel = torch.FloatTensor(kernel1).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt1 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel2).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt2 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel3).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt3 = nn.Parameter(data=kernel, requires_grad=False)

        kernel = torch.FloatTensor(kernel4).unsqueeze(0).unsqueeze(0)
        kernel = np.repeat(kernel, 3, axis=0)
        self.weightt4 = nn.Parameter(data=kernel, requires_grad=False)

        #归一化层   N, C, H, W = 20, 5, 10, 10

        self.GNN1 = model_cls.Net()
        self.GNN2 = model_cls.Net()
        self.GNN3 = model_cls.Net()
        self.GNN4 = model_cls.Net()
        #self.mid_GNN = GNN_model.Net()
        self.GNN_mid = model_cls.Net()

        #self.conv_all = nn.Conv2d(680,340,kernel_size=1)



    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
    def getcoorr(self, bl,boxnum,image1,ba,flow,dc,mi):
        
        coord0 = []
        coord1 = []
        flowgt = []
        dcgt = []
        corrdsbig = coords_grid(1, dc.shape[2], dc.shape[3]).to(bl.device)

        for b in range(ba):
            flowbig = flow[b,:,:,:]
            dcbig = dc[b,:,:,:]
            for l in range(boxnum[b]):
                bminx0 = bl[b, l, 3]
                bminy0 = bl[b, l, 5]
                bmaxx0 = bl[b, l, 2]
                bmaxy0 = bl[b, l, 4]
                blenx0 = bl[b, l, 2] - bl[b, l, 3]
                bleny0 = bl[b, l, 4] - bl[b, l, 5]
                mask1 = (corrdsbig[0,0,:,:]>bminx0) & (corrdsbig[0,0,:,:]<bmaxx0)
                mask2 = (corrdsbig[0, 1, :, :] > bminy0) & (corrdsbig[0, 1, :, :] < bmaxy0)
                maskf = mask2 & mask1
                #检测框的maskf
                maskvalid = ((flowbig[0]!=0)&maskf).detach().cpu().numpy()
                flowuse0 = flowbig[0][maskvalid].unsqueeze(1)
                flowuse1 = flowbig[1][maskvalid].unsqueeze(1)
                flowuse = torch.cat([flowuse0,flowuse1],dim=1)

                dcuse= dcbig[0][maskvalid].unsqueeze(1)

                inds = np.argwhere(maskvalid>0)
                if inds.shape[0] ==0:
                    fs = flowbig.detach().cpu().numpy()
                    plt.imshow(fs[0])
                    ishow = image1.detach().cpu().numpy()
                    plt.imshow(ishow[b,0])
                    plt.show()
                    print('haha')
                useinds = np.random.randint(0, inds.shape[0], mi*mi)
                coords0u = torch.from_numpy(inds[useinds]).permute(1,0).contiguous().view(1,2,mi,mi)
                flowu = flowuse[useinds].permute(1,0).contiguous().view(1,2,mi,mi)
                dcu = dcuse[useinds].permute(1,0).contiguous().view(1,1,mi,mi)
                dcgt.append(dcu)
                flowgt.append(flowu/8)
                coords0 = torch.zeros_like(coords0u)
                coords0[0,0] = coords0u[0,1] 
                coords0[0,1] = coords0u[0,0]
                coords0 = coords0.cuda()
                coord0.append(coords0/8)
                coords1 = coords_grid(1, mi, mi).to(bl.device)
                bminx1 = bl[b, l, 7]
                bminy1 = bl[b, l, 9]
                blenx1 = bl[b, l, 6] - bl[b, l, 7]
                bleny1 = bl[b, l, 8] - bl[b, l, 9]
                coords1[0,0,:,:] = ((coords0[0,0,:,:]-bminx0)/blenx0)*blenx1 + bminx1
                coords1[0, 1, :, :] = ((coords0[0, 1, :, :] - bminy0) / bleny0) * bleny1+bminy1
                coord1.append(coords1/8)

        return  coord0,coord1,flowgt,dcgt
    def getcoorrrandom(self, bl,boxnum,ba,mi):
        
        coord0 = []
        coord1 = []
        corrdsbig = coords_grid(1, 320, 960).to(bl.device)

        for b in range(ba):
            for l in range(boxnum[b]):
                bminx0 = bl[b, l, 3]
                bminy0 = bl[b, l, 5]
                bmaxx0 = bl[b, l, 2]
                bmaxy0 = bl[b, l, 4]
                blenx0 = bl[b, l, 2] - bl[b, l, 3]
                bleny0 = bl[b, l, 4] - bl[b, l, 5]
                mask1 = (corrdsbig[0,0,:,:]>bminx0) & (corrdsbig[0,0,:,:]<bmaxx0)
                mask2 = (corrdsbig[0, 1, :, :] > bminy0) & (corrdsbig[0, 1, :, :] < bmaxy0)
                maskf = mask2 & mask1

                inds = np.argwhere(maskf.detach().cpu().numpy()>0)
                useinds = np.random.randint(0, inds.shape[0], mi*mi)
                coords0u = torch.from_numpy(inds[useinds]).permute(1,0).contiguous().view(1,2,mi,mi)

                coords0 = torch.zeros_like(coords0u)
                coords0[0,0] = coords0u[0,1]
                coords0[0,1] = coords0u[0,0]
                coords0 = coords0.cuda()
                coord0.append(coords0/8)
                coords1 = coords_grid(1, mi, mi).to(bl.device)
                bminx1 = bl[b, l, 7]
                bminy1 = bl[b, l, 9]
                blenx1 = bl[b, l, 6] - bl[b, l, 7]
                bleny1 = bl[b, l, 8] - bl[b, l, 9]
                coords1[0,0,:,:] = ((coords0[0,0,:,:]-bminx0)/blenx0)*blenx1 + bminx1
                coords1[0, 1, :, :] = ((coords0[0, 1, :, :] - bminy0) / bleny0) * bleny1+bminy1
                coord1.append(coords1/8)

        return  coord0,coord1

    def gcorr(self,fmap1, fmap2):
        batch1, dim1, ht1, wd1 = fmap1.shape
        batch2, dim2, ht2, wd2 = fmap2.shape
        fmap1 = fmap1.view(batch1, dim1, ht1 * wd1)
        fmap2 = fmap2.view(batch2, dim2, ht2 * wd2)

        corr = torch.matmul(fmap1.transpose(1, 2), fmap2)
        return corr / torch.sqrt(torch.tensor(dim1).float())


    def forward(self, image1, image2,flow,dc,bl,boxnum,iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """
        now_time = time.time()
        ba, _, h, w = image1.shape
        image1 = 2 * (image1 / 255.0) - 1.0   #归一化处理
        image2 = 2 * (image2 / 255.0) - 1.0
        w1 = self.weightt4
        image21 = F.conv2d(image2, w1, padding = 2, groups = 3)
        image21 = F.interpolate(image21,[int(h*self.delta4), int(w * self.delta4)])
        w2 = self.weightt2
        image23 = F.conv2d(image2, w2, padding=2, groups=3)
        image23 = F.interpolate(image23, [int(h * self.delta2), int(w * self.delta2)]) #0.75
        image26 = F.interpolate(image2, [int(h * 1.25), int(w * 1.25)]) #1.25 
    
        image1 = image1.contiguous()
        image2 = image2.contiguous()
        Fmap1 = []
        Fmap2 = []

        with autocast(enabled=self.args.mixed_precision):
            fmap1 = self.fnet(image1)
            Fmap1.append(fmap1.float())

            #fmap2 = self.fnet(image23)
            fmap2 = self.fnet(image21)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image23)
            Fmap2.append(fmap2.float())

            fmap2 = self.fnet(image2)
            Fmap2.append(fmap2.float())

            #fmap2 = self.fnet(image26)
            fmap2 = self.fnet(image26)
            Fmap2.append(fmap2.float()) 

        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [192, 192], dim=1)
            net = torch.tanh(net)
            inp = torch.tanh(inp)
        corr_fn = CorrpyBlockTTC(Fmap1, Fmap2, net, inp, radius=self.args.corr_radius)

        
        corrd0, corrd1, flow_gt,dc_gt = self.getcoorr(bl, boxnum,image1, ba,flow,dc,10)
        flow_gt = torch.cat([f0 for f0 in flow_gt], dim=0)  #针对车的
        dc_gt = torch.cat([d for d in dc_gt], dim=0)

    

        batch = cnet.shape[0]
        ans = []
        for b in range(batch):
            for i in range(boxnum[b]):
                ans.append(bl[b, i, 1])
        ansall = torch.cat([a.unsqueeze(0) for a in ans], dim=0).unsqueeze(1)


        corrd0 = torch.cat([c0 for c0 in corrd0],dim=0)
        corrd1 = torch.cat([c1 for c1 in corrd1], dim=0)

        flowpu = corrd1 - corrd0  
        expu = torch.ones(sum(boxnum),1,10,10).cuda() + 1

        N = 2

        flow_predictions = []
        exp_predictions = []
        Omid_predictions = []

        loss_mid = []
        last_time = time.time()
        for i in range(N):
            corrd1 = corrd1.detach()
            expu = expu.detach()
            flowpu = flowpu.detach()
        
            corr_feature, corr_net, corr_inp = corr_fn(corrd0, corrd1, bl, ba, boxnum, expu)
  
            corr_f = self.encoder_corr(corr_feature)  # 256
            corr_cont = self.encoder_cont(torch.cat([corr_net, corr_inp], dim=1))  # 128
            motion = self.encoder_mot(torch.cat([flowpu, expu], dim=1))  # 64

            if i == 0:
                GNN_feature = self.GNN1(corrd0.view(sum(boxnum), -1, 100), torch.cat([corr_f, corr_cont, motion], dim=1).view(sum(boxnum), -1, 100))  # 448
            elif i == 1:
                GNN_feature = self.GNN2(corrd0.view(sum(boxnum), -1, 100), torch.cat([corr_f, corr_cont, motion], dim=1).view(sum(boxnum), -1, 100))  # 448
            elif i == 2:
                GNN_feature = self.GNN3(corrd0.view(sum(boxnum), -1, 100), torch.cat([corr_f, corr_cont, motion], dim=1).view(sum(boxnum), -1, 100))  # 448
            else:
                GNN_feature = self.GNN4(corrd0.view(sum(boxnum), -1, 100),
                                    torch.cat([corr_f, corr_cont, motion], dim=1).view(sum(boxnum), -1, 100))  # 448



            d_f = self.flowhead(torch.cat([GNN_feature, flowpu], dim=1))
            dmid = self.midhead(GNN_feature)
            mask = self.maskhead(torch.cat([GNN_feature, corr_inp, flowpu], dim=1))

            corrd1 = corrd1 + d_f
            flowpu = corrd1 - corrd0


            expu = expu + dmid
            Omid = grid2exp(expu) * mask
            Omid = torch.mean(Omid, dim=-1, keepdim=False)
            Omid = torch.mean(Omid, dim=-1, keepdim=False)


            flow_predictions.append(flowpu)
            exp_predictions.append(grid2exp(expu))
            Omid_predictions.append(Omid)
            loss_mid.append((ansall - Omid).abs().mean())

        loss_dc = 0
        loss_flow = 0
        by = 0
        for i in range(N):
            loss_dc = loss_dc+(dc_gt - exp_predictions[i]).abs().mean()
            loss_flow = loss_flow+(flow_predictions[i]-flow_gt).abs().mean()
            by = by+1

        loss = sum(loss_mid)/N

        #return  loss, boxnum, loss_dc/by, loss_flow/by,Omid_predictions[-1]
        return loss, boxnum, loss_dc / by, loss_flow / by