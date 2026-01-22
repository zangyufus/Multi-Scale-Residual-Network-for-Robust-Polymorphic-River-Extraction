import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        BCE_loss = self.criterion(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()
        
class BinaryFocalLoss1(nn.Module): 
    def __init__(self, alpha=1, gamma=2, logits=False, size_average=True, w0 = 1, w1 = 1):
        super(BinaryFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.size_average = size_average
        self.w0 = w0 # the weight of noncrack pixels
        self.w1 = w1 # the weight of crack pixels
        if self.w0 == None or self.w1==None:
            raise ValueError("w must be a number")

    def forward(self, inputs, targets):
        weight=torch.zeros_like(targets)
        weight=torch.fill_(weight, self.w0)
        weight[targets>0]=self.w1
        BCE_loss = nn.BCEWithLogitsLoss(weight=weight, reduction='none')(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.size_average:
            return F_loss.mean()
        else:
            return F_loss.sum()


class MyBCELoss(nn.Module):

    def __init__(self, w0 = None, w1 = None):
        super(MyBCELoss, self).__init__()
        self.w0 = w0 # the weight of noncrack pixels
        self.w1 = w1 # the weight of crack pixels
        if self.w0 == None or self.w1==None:
            raise ValueError("w must be a number")
        
    def forward(self, input, target):
        weight=torch.zeros_like(target)
        weight=torch.fill_(weight, self.w0)
        weight[target>0]=self.w1
        loss=nn.BCELoss(weight=weight,reduction='mean')(input,target)
        return loss

class FocalLossV1(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLossV1, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        Usage is same as nn.BCEWithLogits:
            >>> criteria = FocalLossV1()
            >>> logits = torch.randn(8, 19, 384, 384)
            >>> lbs = torch.randint(0, 2, (8, 19, 384, 384)).float()
            >>> loss = criteria(logits, lbs)
        '''
        probs = torch.sigmoid(logits)
        coeff = torch.abs(label - probs).pow(self.gamma).neg()
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        loss = label * self.alpha * log_probs + (1. - label) * (1. - self.alpha) * log_1_probs
        loss = loss * coeff

        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return 
    
def dice_loss(target,predictive,ep=1e-8):
    intersection = 2 * torch.sum(predictive * target) + ep
    union = torch.sum(predictive) + torch.sum(target) + ep
    loss = 1 - intersection / union
    return loss
 
class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()
 
    def forward(self, logits, targets):
        bs = targets.size(0)
        smooth = 1
        
        probs = F.sigmoid(logits)
        m1 = probs.view(bs, -1)
        m2 = targets.view(bs, -1)
        intersection = (m1 * m2)
 
        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / bs
        return score

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)      
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        weight=torch.zeros_like(targets)
        weight=torch.fill_(weight, 0.04)
        weight[targets>0]=0.96
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean', weight=weight)
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE




class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, smooth=1, lambda_dice=1, lambda_focal=1):

        super(DiceFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.lambda_dice = lambda_dice
        self.lambda_focal = lambda_focal
        self.bce_loss = nn.BCEWithLogitsLoss(reduction="none")  # 用 BCE 计算 Focal Loss

    def forward(self, inputs, targets):

        # Sigmoid 归一化
        inputs = torch.sigmoid(inputs)

        # 展平处理
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # ---- 计算 Dice Loss ----
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)

        # ---- 计算 Focal Loss ----
        bce = self.bce_loss(inputs, targets)  # 计算 BCE Loss
        pt = torch.exp(-bce)  # pt = exp(-BCE) 表示预测正确的概率
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * bce).mean()

        # ---- 组合损失 ----
        total_loss = self.lambda_dice * dice_loss + self.lambda_focal * focal_loss

        return total_loss


class TverskyFocalLoss(nn.Module):
    def __init__(self, alpha=0.2, beta=0.8, gamma=2.0, smooth=1e-6):
        super(TverskyFocalLoss, self).__init__()
        self.alpha = alpha 
        self.beta = beta 
        self.gamma = gamma 
        self.smooth = smooth

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        tversky_loss = 1 - (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        weight = targets * 0.8 + 0.2 
        BCE = F.binary_cross_entropy(inputs, targets, weight=weight)

        focal_loss = (1 - inputs) ** self.gamma * BCE

        loss = tversky_loss + focal_loss.mean()

        return loss


class TverskyDiceLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, lambda_t=1):
        super(TverskyDiceLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.lambda_t = lambda_t

    def forward(self, inputs, targets):

        inputs = torch.sigmoid(inputs)

        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()
        dice_loss = 1 - (2 * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky_loss = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        total_loss = dice_loss + self.lambda_t * (1 - tversky_loss)
        
        return total_loss

class TverskyBCELoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1, lambda_t=0.5, lambda_bce=0.5):

        super(TverskyBCELoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.lambda_t = lambda_t
        self.lambda_bce = lambda_bce
        self.bce_loss = nn.BCEWithLogitsLoss()  # BCE Loss 计算（带 Logits）

    def forward(self, inputs, targets):

        # Sigmoid 归一化
        inputs = torch.sigmoid(inputs)

        # 展平处理
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # 计算交叉熵损失 BCE Loss
        bce_loss = self.bce_loss(inputs, targets)

        # 计算 Tversky Loss
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        tversky_loss = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # 组合损失
        total_loss = self.lambda_bce * bce_loss + self.lambda_t * (1 - tversky_loss)

        return total_loss

class ExpLogDiceLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=1e-6):
        super(ExpLogDiceLoss, self).__init__()
        self.gamma = gamma
        self.smooth = smooth  # 避免分母为0

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)  # 将 logits 转换为概率
        inputs = inputs.view(-1)  # 展平
        targets = targets.view(-1)

        # 计算 TP, FP, FN
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        # 计算 Dice Coefficient
        dice_score = (2 * TP + self.smooth) / (2 * TP + FP + FN + self.smooth)
        
        # 计算 Exp-Log-Dice Loss
        loss = -torch.log(1 - (1 - dice_score) ** self.gamma + self.smooth)

        return loss