import torch
import torch.nn as nn
from torch.autograd import Function
import torch.nn.functional as F
import numpy as np

class Discriminator(nn.Module):
    def __init__(self):
        # 域判别器结构
        super(Discriminator, self).__init__()
        layers = [
            nn.ReLU(),
            nn.Linear(16, 4),
            nn.ReLU(),
            nn.Linear(4, 2),
            nn.Softmax(dim=1)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class UncertainLoss(nn.Module):
    def __init__(self) -> None:
        super(UncertainLoss,self).__init__()
    
    def forward(self, logits, target, uncertainty):
        logits = logits + 1e-10
        l = torch.log(logits)
        u = logits * l
        u = (u.sum(dim=1)).neg()
        loss = torch.nan_to_num(u,0.0)

        un_loss = loss * torch.exp(uncertainty.neg())
        un_loss = torch.nan_to_num_(un_loss, 0.0)
        loss =loss + un_loss

        return loss.mean()
    
class ReverseLayer(Function):
    # 梯度反转层
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class LambdaSheduler(nn.Module):
    def __init__(self, gamma=1.0, max_iter=1000):
        super(LambdaSheduler, self).__init__()
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0

    def lamb(self):
        p = self.curr_iter / self.max_iter
        lamb = 2. / (1. + np.exp(-self.gamma * p)) - 1
        return lamb
    
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class DAANloss(LambdaSheduler):
    def __init__(self):
        super(DAANloss, self).__init__()
        # 全局域判别器
        self.domain_classfier = Discriminator()
        # 域对抗损失函数
        self.loss_fn = UncertainLoss()
        # 局部域判别器
        self.local_classifiers = nn.ModuleList()
        for _ in range(2):
            self.local_classifiers.append(Discriminator())

        self.glo_dis = 0
        self.loc_dis = 0
        self.dynamic_factor = 0.5
      
    def forward(self, source, target, source_logits, target_logits):
        lamb = self.lamb()
        self.step()
        
        # 分类不确定性
        source_un = self.get_uncertainty(source_logits)
        target_un = self.get_uncertainty(target_logits)

        # 全局和局部域对抗损失
        source_loss_g = self.glo_adv(source, source_un, True, lamb)
        target_loss_g = self.glo_adv(target, target_un, False, lamb)
        source_loss_l = self.loc_adv(source, source_logits, source_un, True, lamb)
        target_loss_l = self.loc_adv(target, target_logits, target_un, False, lamb)

        global_loss = 0.5 * (source_loss_g + target_loss_g)*0.1
        local_loss = 0.5 * (source_loss_l + target_loss_l)*0.05

        # 计算A-距离
        self.glo_dis = self.glo_dis + 2 * (1 - 2 * global_loss.cpu().item())
        self.loc_dis = self.loc_dis + 2 * (1 - 2 * (local_loss / 2).cpu().item())

        # 加权得到域对抗损失
        adv_loss = (1 - self.dynamic_factor) * global_loss + self.dynamic_factor * local_loss

        return adv_loss

    def get_uncertainty(self, clf):
        l = torch.log(clf)
        u = clf * l
        u = (u.sum(dim=1)).neg()
        u = torch.nan_to_num(u,0.0)
        return u
    
    def glo_adv(self, x, uncertainty, source=True, lamb=1.0):
        x = ReverseLayer.apply(x, lamb)
        domain_pred = self.domain_classfier(x)
        if source == True:
            domain = torch.ones_like(domain_pred)
        else:
            domain = torch.zeros_like(domain_pred)
        loss_adv = self.loss_fn(domain_pred, domain, uncertainty)
        return loss_adv

    def loc_adv(self, x, logits, uncertainty,source=True, lamb = 1.0):
        x = ReverseLayer.apply(x, lamb)
        loss_adv = 0.0
        for c in range(2):
            logits_c = logits[:, c].reshape((logits.shape[0], 1))
            features_c = logits_c * x
            domain_pred = self.local_classifiers[c](features_c)
            if source == True:
                domain = torch.ones_like(domain_pred)
            else:
                domain = torch.zeros_like(domain_pred)
            loss_adv = loss_adv + self.loss_fn(domain_pred, domain, uncertainty)
        return loss_adv

    def update_dynamic(self, epoch_length):
        # 更新动态对抗因子
        if self.glo_dis == 0 and self.loc_dis == 0:
            self.dynamic_factor = 0.5
        else:
            self.glo_dis = self.glo_dis / epoch_length
            self.loc_dis = self.loc_dis / epoch_length
            self.dynamic_factor = 1 - self.glo_dis / (self.glo_dis + self.loc_dis)
        self.glo_dis = 0
        self.loc_dis = 0

