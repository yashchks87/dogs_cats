import torch
import torch.nn as nn
import torch.nn.functional as F
def calculate_precision(inputs, targets):
    """
        Calculate precision
        Precision = TP / (TP + FP)
    """
    inputs = F.sigmoid(inputs)
    inputs = torch.where(inputs > 0.5, 1.0, 0.0)
    tp = torch.logical_and((inputs == 1.0), (targets == 1.0)).sum().float()
    fp = torch.logical_and((inputs == 1.0), (targets == 0.0)).sum().float()
    precision = tp / (tp + fp + 1e-8)
    return precision

def calculate_recall(inputs, targets):
    """
        Calculate precision
        Precision = TP / (TP + FN)
    """
    inputs = F.sigmoid(inputs)
    inputs = torch.where(inputs > 0.5, 1.0, 0.0)
    tp = torch.logical_and((inputs == 1.0), (targets == 1.0)).sum().float()
    fn = torch.logical_and((inputs == 0.0), (targets == 1.0)).sum().float()
    recall = tp / (tp + fn + 1e-8)
    return recall

def calculate_cf(inputs, targets):
    inputs = F.sigmoid(inputs)
    inputs = torch.where(inputs > 0.5, 1.0, 0.0)
    tp = torch.logical_and((inputs == 1.0), (targets == 1.0)).sum().float()
    fn = torch.logical_and((inputs == 0.0), (targets == 1.0)).sum().float()
    fp = torch.logical_and((inputs == 1.0), (targets == 0.0)).sum().float()
    tn = torch.logical_and((inputs == 0.0), (targets == 0.0)).sum().float()
    return tp, fn, fp, tn