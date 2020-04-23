import math
import torch
from torch import nn


class WingLoss(nn.Module):
    def __init__(self, omega=1e-3, epsilon=0.5):
        super(WingLoss, self).__init__()
        self.omega=omega
        self.epsilon=epsilon

    def forward(self,pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        #print(delta_y.shape)
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        #print("delta_y1 delta_y2 shape: ",delta_y1.shape, delta_y2.shape)
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        #print("loss2: ",loss2)
        #print("delta_y2: ",loss2)
        #print("C: ",C)
        #print("loss1 and loss2 shape: ",loss1.shape, loss2.shape)
        #return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=1e-3, theta=0.5, epsilon=1, alpha=0.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH
        :param target: BxNxHxH
        :return:
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))
'''
class lemloss(nn.module):
    def __init__(self, gamma=1):
        super(lemloss, self).__init__()
        self.gamma=gamma

    def forward(self,pred, target):
        y = target
        y_hat = pred
        logy = torch.log(target.abs())
        logy_hat = torch.log(pred.abs())
        delta_y = torch.exp(self.gamma*(y - y_hat).abs())
        loss = torch.mean(delta_y) 
        return loss

class LEMLoss(nn.Module):
    def __init__(self):
        super(LEMLoss, self).__init__()

    def forward(self,pred, target):
        y = target
        y_hat = pred
#        logy = torch.log(target.abs())
#        logy_hat = torch.log(pred.abs())
#        logy = (torch.log(y+1)/(torch.log(1+torch.max(y))))
        logy = torch.log(y+1)
#        logy_hat = (torch.log(y_hat+1)/(torch.log(1+torch.max(y_hat))))
        logy_hat = torch.log(y_hat+1)
        gamma = ((logy-logy_hat).abs())/2
        exponent = torch.div((logy - logy_hat).abs(),gamma)
        delta_y = torch.exp(exponent)
        loss = torch.mean(delta_y) 
        return loss

'''

class LEMLoss(nn.Module):
    def __init__(self):
        super(LEMLoss, self).__init__()

    def forward(self,pred, target):
        y = target
        y_hat = pred
#         logy = torch.log(target.abs())
#         logy_hat = torch.log(pred.abs())
        #print(torch.min(y),torch.max(y))
        #print(torch.min(y_hat),torch.max(y_hat))
        logy = (torch.log(y+1)/(torch.log(1+torch.max(y))))
#         logy = torch.log(y+1)
        logy_hat = (torch.log(y_hat+1)/(torch.log(1+torch.max(y_hat))))
        #print(torch.min(logy),torch.max(logy))
        #print(torch.min(logy_hat),torch.max(logy_hat))
#         logy_hat = torch.log(y_hat+1)
        gamma = ((logy-logy_hat).abs())/2
        #print(torch.min(gamma),torch.max(gamma))
        exponent = torch.div((logy - logy_hat).abs(),gamma+1e-3)
        delta_y = torch.exp(exponent)
        loss = torch.mean(delta_y)
        return loss
'''

if __name__ == "__main__":
    #loss_func = AdaptiveWingLoss()
    loss_func = WingLoss()
    #loss_func = LEMLoss()
    y = torch.rand(2, 68, 64, 64)
    y_hat = torch.rand(2, 68, 64, 64)
    y_hat.requires_grad_(True)
    loss = loss_func(y_hat, y)
    loss.backward()
    print(loss)
'''


