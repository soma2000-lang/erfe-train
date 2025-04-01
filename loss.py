import torch
import torch.nn as nn
import torch.nn.functional as F


class SegmentationLoss(nn.Module):
    def __init__(self, penalty_rate=2.0):
        super(SegmentationLoss, self).__init__()
        self.penalty_rate = penalty_rate
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, y_pred, y_true):
  
        if not isinstance(y_pred, torch.Tensor):
            raise TypeError(f"Expected y_pred to be a Tensor, but got {type(y_pred).__name__}")
            
        runway_true = y_true[:, 0, :, :]  # Class 1 - Runway area
        aiming_point_true = y_true[:, 1, :, :]  # Class 2 - Aiming point
        threshold_true = y_true[:, 2 :, :]  # Class 3 - Threshold marking
        
 
        weights = torch.ones_like(runway_true)
        weights = weights + (self.penalty_rate - 1) * aiming_point_true
        weights = weights + (self.penalty_rate - 1) * threshold_true
        
        y_true_indices = torch.argmax(y_true, dim=1)
        

        loss = self.ce_loss(y_pred, y_true_indices)
        weighted_loss = loss * weights
        
        return torch.mean(weighted_loss)

class FeatureLineLoss(nn.Module):
    def __init__(self):
        super(FeatureLineLoss, self).__init__()
    
    def forward(self, y_pred, y_true):
        epsilon = 1e-7
        y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
        loss = -y_true * torch.log(y_pred)
        return torch.mean(loss)


class CombinedLoss(nn.Module):
    def __init__(self, penalty_rate=2.0):
        super(CombinedLoss, self).__init__()
        self.seg_loss = SegmentationLoss(penalty_rate)
        
    def forward(self, y_pred, y_true):

        seg_pred = y_pred  
        seg_true = y_true  
        

        if not isinstance(seg_pred, torch.Tensor):
            raise TypeError(f"Expected seg_pred to be a tensor, got {type(seg_pred)}")
            
      
        seg_loss_val = self.seg_loss(seg_pred, seg_true)
        
        return seg_loss_val
