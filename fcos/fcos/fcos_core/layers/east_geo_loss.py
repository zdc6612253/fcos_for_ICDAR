import torch
from torch import nn

class EastGeoLoss(nn.Module):
    def __init__(self, beta=1./9, size_average=True):
        super(EastGeoLoss, self).__init__()
        self.beta = beta
        self.size_average = size_average
    
    def forward(self, input, target, weight=None):
        if target.numel() == 0:
            return torch.tensor(0.).to(input.device)
        input = input.reshape(-1,4,2)
        target = target.reshape(-1,4,2)
        # remove box with minsize <= 1e-4
        target1 = torch.roll(target, 1, dims=1)  # n*4*2
        edge_vec = (target1 - target).float()
        edge_length, _ = (edge_vec ** 2).sum(dim=-1).sqrt().min(dim=-1)  # shape:n 表示差距
        valid_mask = edge_length > 1e-4
        if valid_mask.sum() < 1:
            return torch.tensor(0.).to(input.device)

        input = input.reshape(input.shape[0],4,2)  # n*4*2
        input1 = input.view(-1, 1, target.size(1), 2)  # n*1*4*2
        target1 = [target.roll(i, dims=1) for i in range(target.size(1))]  # n*4*4*2
        target1 = torch.stack(target1, dim=1)  # n*4*4*2
        input1 = input1[valid_mask]  # 第一个维度的mask，筛选出合格的样本，n*4*4*2
        target1 = target1[valid_mask]

        diff = torch.abs(input1 - target1)
        loss = torch.where(diff < self.beta, 0.5 * diff ** 2 / self.beta, diff - 0.5 * self.beta)  # n*4*4*2
        loss = loss.view(-1, target.size(1), target.size(1) * 2)  # n*4*8
        loss = torch.sum(loss, dim=-1)  # 对于坐标的16个数值求和
        loss, _ = torch.min(loss, dim=1)  # 找到最小的旋转loss
        
        avg_factor = target.size(1) * 2
        loss = loss / avg_factor
        if weight is not None:
            weight = weight[valid_mask]
            loss = loss * weight
            loss = loss.sum() / weight.sum()
        elif self.size_average:
            return loss.mean()
        return loss.sum()
