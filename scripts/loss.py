import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_bce(inputs, targets, smooth = 1e-8):
    i = F.sigmoid(inputs)
    # bce = -((targets * torch.log(i + smooth)) + ((1 - targets) * torch.log(1 - i + smooth))).mean()
    bce = nn.BCELoss(reduction='mean')
    return bce(i, targets)
    # return bce