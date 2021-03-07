import torch
import torch.nn as nn
import numpy as np

    
class DepthLoss(nn.Module):
    def __init__(self):
        super(DepthLoss,self).__init__()
        self.mse = nn.MSELoss()
        self.grad_factor = 10.
        self.normal_factor = 1.
    def forward(self,criterion,pred,target,epoch=0):
        if (criterion == 'l1'):
            depth_loss = self.L1_imp_Loss(pred,target)
        elif (criterion =='l2'):
            depth_loss = self.L2_imp_Loss(pred,target)
        else:
            depth_loss = self.RMSLELoss(pred,target)
        grad_pred, grad_target = self.imgrad_yx(pred), self.imgrad_yx(target)
        grad_loss = self.GradLoss(grad_pred,grad_target)*self.grad_factor * (epoch>3)
        normal_loss = self.NormLoss(grad_pred,grad_target)*self.normal_factor * (epoch>7)
        return depth_loss + grad_loss+ normal_loss
    
    def GradLoss(self,grad_target,grad_pred):
        return torch.sum( torch.mean( torch.abs(grad_target-grad_pred) ) )
    
    def NormLoss(self, grad_target, grad_pred):
        prod = ( grad_pred[:,:,None,:] @ grad_target[:,:,:,None] ).squeeze(-1).squeeze(-1)
        pred_norm = torch.sqrt( torch.sum( grad_pred**2, dim=-1 ) )
        target_norm = torch.sqrt( torch.sum( grad_target**2, dim=-1 ) ) 
        return 1 - torch.mean( prod/(pred_norm*target_norm) )
    
    def RMSLELoss(self, pred, target):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(target + 1)))
    
    def L1_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
    
    def L2_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss
    
    def imgrad_yx(self,img):
        N,C,_,_ = img.size()
        grad_y, grad_x = self.imgrad(img)
        return torch.cat((grad_y.view(N,C,-1), grad_x.view(N,C,-1)), dim=1)
    
    def imgrad(self,img):
        img = torch.mean(img, 1, True)
        fx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv1.weight = nn.Parameter(weight)
        grad_x = conv1(img)

        fy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
        if img.is_cuda:
            weight = weight.cuda()
        conv2.weight = nn.Parameter(weight)
        grad_y = conv2(img)
        return grad_y, grad_x
    

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight, device):
        super(CrossEntropyLoss2d, self).__init__()
        if device:
            self.weight = torch.tensor(weight).cuda()
        else:
            self.weight = torch.tensor(weight)
        self.num_classes = len(self.weight) + 1  # +1 for void
        if self.num_classes < 2**8:
            self.dtype = torch.uint8
        else:
            self.dtype = torch.int16
        self.ce_loss = nn.CrossEntropyLoss(
            torch.from_numpy(np.array(weight)).float(),
            reduction='none',
            ignore_index=-1
        )
        if device:
            self.ce_loss.cuda()
    def forward(self, inputs_scales, targets_scales):
        losses = []
        for inputs, targets in zip(inputs_scales, targets_scales):
            # mask = targets > 0
            targets_m = targets.clone()
            targets_m -= 1
            loss_all = self.ce_loss(inputs, targets_m.long())

            number_of_pixels_per_class = \
                torch.bincount(targets.flatten().type(self.dtype),
                               minlength=self.num_classes)
            divisor_weighted_pixel_sum = \
                torch.sum(number_of_pixels_per_class[1:] * self.weight)   # without void
            losses.append(torch.sum(loss_all) / divisor_weighted_pixel_sum)
            # losses.append(torch.sum(loss_all) / torch.sum(mask.float()))

        return sum(losses)

    def L1_imp_Loss(self, pred, target):
        valid_mask = (target>0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss
