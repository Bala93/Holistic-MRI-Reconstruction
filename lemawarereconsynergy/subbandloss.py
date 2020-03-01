import torch
import torch.nn as nn
import numpy as np
import math
import pywt
from torch.autograd import Variable
def get_upsample_filter(size):
    """Make a 2D bilinear kernel suitable for upsampling"""
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    filter = (1 - abs(og[0] - center) / factor) * \
             (1 - abs(og[1] - center) / factor)
    return torch.from_numpy(filter).float()

        
class L1_Charbonnier_loss(nn.Module):
    """L1 Charbonnierloss."""
    def __init__(self):
        super(L1_Charbonnier_loss, self).__init__()
        self.eps = 1e-6

    def forward(self, X, Y):
        diff = torch.add(X, -Y)
        error = torch.sqrt( diff * diff + self.eps )
        loss = torch.sum(error) 
        #loss = torch.mean(error) 
        return loss


class subbandEnergy_loss(nn.Module):
    """Loss in DWT subbands."""
    def __init__(self):
        print("inside init of sub band")
        super(subbandEnergy_loss, self).__init__()
        self.w = pywt.Wavelet('db1')
        #self.w = pywt.Wavelet('db2')
        self.dec_hi = torch.Tensor(self.w.dec_hi[::-1]) 
        self.dec_lo = torch.Tensor(self.w.dec_lo[::-1])
        self.rec_hi = torch.Tensor(self.w.rec_hi)
        self.rec_lo = torch.Tensor(self.w.rec_lo)
        self.filters = torch.stack([self.dec_lo.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                       self.dec_lo.unsqueeze(0)*self.dec_hi.unsqueeze(1),
                       self.dec_hi.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                       self.dec_hi.unsqueeze(0)*self.dec_hi.unsqueeze(1)], dim=0)
        
        self.filters = self.filters.cuda()
        self.filters = Variable(self.filters[:,None])
        self.subband_loss = nn.MSELoss()
        
        # self.padded = torch.nn.functional.pad(vimg,(0,0,0,0))

    def forward(self, X, Y):
        #x - Hr_2x
        #y - label
        h,w = X.size(2),X.size(3)
        #print(X.size(),Y.size())
        #print("X max: ",torch.max(X.data), "X min: ",torch.min(X.data))
        #print("Y max: ",torch.max(Y.data), "Y min: ",torch.min(Y.data))
        res1 = torch.nn.functional.conv2d(X,self.filters,stride=2)
        res1_LL = res1[:,0,:,:]
        res1_LH = res1[:,1,:,:]
        res1_HL = res1[:,2,:,:]
        res1_HH = res1[:,3,:,:]
        res2 = torch.nn.functional.conv2d(Y,self.filters,stride=2)
        res2_LL = res2[:,0,:,:]
        res2_LH = res2[:,1,:,:]
        res2_HL = res2[:,2,:,:]
        res2_HH = res2[:,3,:,:]
        #print(res2_LH.size())
        
        loss_LL = torch.sqrt(torch.sum((res1_LL-res2_LL)**2)) 
        loss_LH = torch.sqrt(torch.sum((res1_LH-res2_LH)**2)) 
        loss_HL = torch.sqrt(torch.sum((res1_HL-res2_HL)**2)) 
        loss_HH = torch.sqrt(torch.sum((res1_HH-res2_HH)**2)) 
        
        #loss = (0.33 * loss_LH)+(0.33 * loss_HL)+(0.34 * loss_HH)	
        #loss = (0.01*loss_LL)+(1 * loss_LH)+(1 * loss_HL)+(1 * loss_HH)
        margin = 0
        alpha = 1.0
        loss = (1 * loss_LH)+(1 * loss_HL)+(1 * loss_HH)	

        res1_LH_square = res1_LH ** 2 #batchsize x 32 x32
        #print(res1_LH_square.shape)
        vectorshape=res1_LH_square.shape[1]*res1_LH_square.shape[2]
        res1_LH_square = res1_LH_square.view(-1,vectorshape) # batchsizex1024
        res1_LH_square_sum = torch.sum(res1_LH_square,dim=1)#batchsize

        res2_LH_square = res2_LH ** 2 #batchsize x 32 x32
        res2_LH_square = res2_LH_square.view(-1,vectorshape) # batchsizex1024
        res2_LH_square_sum = torch.sum(res2_LH_square,dim=1)#batchsize
        #print("res1_LH max: ", torch.max(res1_LH.data),"res1_LH min: ", torch.min(res1_LH.data)) 
        #print("res2_LH max: ", torch.max(res2_LH.data),"res2_LH min: ", torch.min(res2_LH.data)) 
        #print("res1_LH_square: ", torch.mean(res1_LH_square_sum.data),"res2_LH_square: ", torch.mean(res2_LH_square_sum.data))
        mean_energy_diff_LH = ((res2_LH_square)*alpha) - res1_LH_square
        loss_tex_LH = torch.mean(nn.functional.relu(mean_energy_diff_LH + margin))
        #print(mean_energy_diff_LH)

        res1_HL_square = res1_HL ** 2 #batchsize x 32 x32
        res1_HL_square = res1_HL_square.view(-1,vectorshape) # batchsizex1024
        res1_HL_square_sum = torch.sum(res1_HL_square,dim=1)#batchsizex1

        res2_HL_square = res2_HL ** 2 #batchsize x 32 x32
        res2_HL_square = res2_HL_square.view(-1,vectorshape) # batchsizex1024
        res2_HL_square_sum = torch.sum(res2_HL_square,dim=1)#batchsizex1

        mean_energy_diff_HL = ((res2_HL_square)*alpha) - res1_HL_square
        loss_tex_HL = torch.mean(nn.functional.relu(mean_energy_diff_HL + margin))
        #print(mean_energy_diff_LH)

        res1_HH_square = res1_HH ** 2 #batchsize x 32 x32
        res1_HH_square = res1_HH_square.view(-1,vectorshape) # batchsizex1024
        res1_HH_square_sum = torch.sum(res1_HH_square,dim=1)#batchsizex1    

        res2_HH_square = res2_HH ** 2 #batchsize x 32 x32
        res2_HH_square = res2_HH_square.view(-1,vectorshape) # batchsizex1024
        res2_HH_square_sum = torch.sum(res2_HH_square,dim=1)#batchsizex1    

        mean_energy_diff_HH = ((res2_HH_square)*alpha) - res1_HH_square
        loss_tex_HH = torch.mean(nn.functional.relu(mean_energy_diff_HH + margin))
        
        loss_tex = loss_tex_LH + loss_tex_HL + loss_tex_HH

        #return loss,loss_LL,loss_LH,loss_HL,loss_HH, loss_tex
        return loss_tex
'''
class subbandEnergy_loss(nn.Module):
    """Loss in DWT subbands."""
    def __init__(self):
        print("inside init of sub band")
        super(subbandEnergy_loss, self).__init__()
        self.w = pywt.Wavelet('db1')
        #self.w = pywt.Wavelet('db2')
        self.dec_hi = torch.Tensor(self.w.dec_hi[::-1]) 
        self.dec_lo = torch.Tensor(self.w.dec_lo[::-1])
        self.rec_hi = torch.Tensor(self.w.rec_hi)
        self.rec_lo = torch.Tensor(self.w.rec_lo)
        self.filters = torch.stack([self.dec_lo.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                       self.dec_lo.unsqueeze(0)*self.dec_hi.unsqueeze(1),
                       self.dec_hi.unsqueeze(0)*self.dec_lo.unsqueeze(1),
                       self.dec_hi.unsqueeze(0)*self.dec_hi.unsqueeze(1)], dim=0)
        
        self.filters = self.filters.cuda()
        self.filters = Variable(self.filters[:,None])
        self.subband_loss = nn.MSELoss()
        
        # self.padded = torch.nn.functional.pad(vimg,(0,0,0,0))

    def forward(self, X, Y):
        #x - Hr_2x
        #y - label
        h,w = X.size(2),X.size(3)
        #print(X.size(),Y.size())
        #print("X max: ",torch.max(X.data), "X min: ",torch.min(X.data))
        #print("Y max: ",torch.max(Y.data), "Y min: ",torch.min(Y.data))
        res1 = torch.nn.functional.conv2d(X,self.filters,stride=2)
        res1_LL = res1[:,0,:,:]
        res1_LH = res1[:,1,:,:]
        res1_HL = res1[:,2,:,:]
        res1_HH = res1[:,3,:,:]
        res2 = torch.nn.functional.conv2d(Y,self.filters,stride=2)
        res2_LL = res2[:,0,:,:]
        res2_LH = res2[:,1,:,:]
        res2_HL = res2[:,2,:,:]
        res2_HH = res2[:,3,:,:]
        #print(res2_LH.size())
        
        loss_LL = torch.sqrt(torch.sum((res1_LL-res2_LL)**2)) 
        loss_LH = torch.sqrt(torch.sum((res1_LH-res2_LH)**2)) 
        loss_HL = torch.sqrt(torch.sum((res1_HL-res2_HL)**2)) 
        loss_HH = torch.sqrt(torch.sum((res1_HH-res2_HH)**2)) 
        
        #loss = (0.33 * loss_LH)+(0.33 * loss_HL)+(0.34 * loss_HH)	
        #loss = (0.01*loss_LL)+(1 * loss_LH)+(1 * loss_HL)+(1 * loss_HH)
        margin = 0
        alpha = 1.0
        loss = (1 * loss_LH)+(1 * loss_HL)+(1 * loss_HH)	

        res1_LH_square = res1_LH ** 2 #batchsize x 32 x32
        #print(res1_LH_square.shape)
        vectorshape=res1_LH_square.shape[1]*res1_LH_square.shape[2]
        res1_LH_square = res1_LH_square.view(-1,vectorshape) # batchsizex1024
        res1_LH_square_sum = torch.sum(res1_LH_square,dim=1)#batchsize

        res2_LH_square = res2_LH ** 2 #batchsize x 32 x32
        res2_LH_square = res2_LH_square.view(-1,vectorshape) # batchsizex1024
        res2_LH_square_sum = torch.sum(res2_LH_square,dim=1)#batchsize
        #print("res1_LH max: ", torch.max(res1_LH.data),"res1_LH min: ", torch.min(res1_LH.data)) 
        #print("res2_LH max: ", torch.max(res2_LH.data),"res2_LH min: ", torch.min(res2_LH.data)) 
        #print("res1_LH_square: ", torch.mean(res1_LH_square_sum.data),"res2_LH_square: ", torch.mean(res2_LH_square_sum.data))
        mean_energy_diff_LH = (torch.mean(res2_LH_square_sum)*alpha) - torch.mean(res1_LH_square_sum)
        loss_tex_LH = nn.functional.relu(mean_energy_diff_LH + margin)
        #print(mean_energy_diff_LH)

        res1_HL_square = res1_HL ** 2 #batchsize x 32 x32
        res1_HL_square = res1_HL_square.view(-1,vectorshape) # batchsizex1024
        res1_HL_square_sum = torch.sum(res1_HL_square,dim=1)#batchsizex1

        res2_HL_square = res2_HL ** 2 #batchsize x 32 x32
        res2_HL_square = res2_HL_square.view(-1,vectorshape) # batchsizex1024
        res2_HL_square_sum = torch.sum(res2_HL_square,dim=1)#batchsizex1

        mean_energy_diff_HL = (torch.mean(res2_HL_square_sum)*alpha) - torch.mean(res1_HL_square_sum)
        loss_tex_HL = nn.functional.relu(mean_energy_diff_HL + margin)

        res1_HH_square = res1_HH ** 2 #batchsize x 32 x32
        res1_HH_square = res1_HH_square.view(-1,vectorshape) # batchsizex1024
        res1_HH_square_sum = torch.sum(res1_HH_square,dim=1)#batchsizex1    

        res2_HH_square = res2_HH ** 2 #batchsize x 32 x32
        res2_HH_square = res2_HH_square.view(-1,vectorshape) # batchsizex1024
        res2_HH_square_sum = torch.sum(res2_HH_square,dim=1)#batchsizex1    

        mean_energy_diff_HH = (torch.mean(res2_HH_square_sum)*alpha) - torch.mean(res1_HH_square_sum)
        loss_tex_HH = nn.functional.relu(mean_energy_diff_HH + margin)
        
        loss_tex = loss_tex_LH + loss_tex_HL + loss_tex_HH

        #return loss,loss_LL,loss_LH,loss_HL,loss_HH, loss_tex
        return loss_tex
'''
