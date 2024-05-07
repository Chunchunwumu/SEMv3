
import torch
from torch import nn
import torch.nn.functional as F
from .utils import cal_token_cm
import numpy as np

def sigmoid_focal_loss_mutil( pred,
                        target,
                        weight=1.0,
                        gamma=2.0,
                        alpha=0.25,
                        reduction='mean'):

    pred_sigmoid = pred.sigmoid()
    target = F.one_hot( target, 4 ).permute(2,0,1).float()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def sigmoid_focal_loss_mutil_2( pred,
                        target,
                        weight=1.0,
                        gamma=2.0,
                        alpha=0.25,
                        reduction='mean'):

    pred_sigmoid = pred.sigmoid()
    target = F.one_hot( target, 2 ).permute(2,0,1).float()
    pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    weight = (alpha * target + (1 - alpha) * (1 - target)) * weight
    weight = weight * pt.pow(gamma)
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none') * weight
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

def dice_loss(pred, target, reduction='mean'):
    pred = pred.sigmoid()
    target = target.type_as(pred)

    a = pred * target # |X?Y|
    b = pred * pred + 1e-3  # |X|
    c = target * target + 1e-3  # |Y|
    d = (2 * a) / (b + c)
    loss = 1 - d
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()

class MergeHead_new_token_v1(nn.Module):
    def __init__(self, in_channels, num_kernel_layers, loss):
        super().__init__()
        
        self.merge_center = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels//2,1,1),
            nn.ReLU(),
            
        )
        self.merge_center_pred = nn.Conv2d(in_channels//2, 2,1,1)
        self.merge = nn.Sequential(
            nn.Conv2d(in_channels+in_channels//2, in_channels, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, 3,1,1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels//2,1,1),
            nn.ReLU(),
            nn.Conv2d(in_channels//2, 4,1,1)
        )
        self.loss_factor = loss['factor']

    def forward(self, feats, masks, layouts):
        B, C, _, _ = feats.shape
        center_feature = self.merge_center(feats)
        center_pred = self.merge_center_pred(center_feature)

        segments_logits_merge = self.merge( torch.cat([feats,center_feature], dim=1) )
        segments_logits = list()
        for batch_idx in range(B):
            
            segments_logit = segments_logits_merge[batch_idx].squeeze(0) # (N, H, W)
            segments_logit = segments_logit.masked_fill((1-masks[batch_idx])[None,:,:].to(torch.bool), float(-1e8)) # remove padding pixels
            segments_logits.append(segments_logit)
        segments_logits = torch.stack(segments_logits, dim=0)

        result_info = dict()
        if self.training:
            layouts_center = torch.where( layouts==0, 1,0 ).to(layouts)
            batch_loss = []
            batch_center_loss = []

            for batch_idx in range(segments_logits.size(0)):
                # cal merge loss
                index = torch.nonzero(masks[batch_idx])
                end_col = index[-1][1]
                end_row = index[-1][0]
                segments_loss = sigmoid_focal_loss_mutil(
                    segments_logits[batch_idx][:, :end_row+1, :end_col+1], # (N, H, W)
                    layouts[batch_idx][:end_row+1, :end_col+1].to(torch.long), # (N, H, W)
                    reduction='none',
                )
                segments_center_loss = sigmoid_focal_loss_mutil_2(
                    center_pred[batch_idx][:, :end_row+1, :end_col+1], # (N, H, W)
                    layouts_center[batch_idx][:end_row+1, :end_col+1].to(torch.long), # (N, H, W)
                    reduction='none',
                )

                grid_num = int(masks[batch_idx].sum())
                
                segments_loss = self.loss_factor * (segments_loss).sum() / (grid_num + 1e-5)
                segments_center_loss = self.loss_factor * (segments_center_loss).sum() / (grid_num + 1e-5)
                
                batch_loss.append(segments_loss)
                batch_center_loss.append(segments_center_loss)
            confusion_matrix = cal_token_cm( segments_logits.detach().float(), layouts, masks )
            acc = (confusion_matrix*np.eye(4)).sum()/(confusion_matrix.sum()+1e-5)
            result_info = dict(loss=sum(batch_loss)/len(batch_loss), c_loss=sum(batch_center_loss)/len(batch_center_loss), acc=acc)

        return result_info, segments_logits


def build_merge_head_new_token_v1(cfg):
    merge_head = MergeHead_new_token_v1(
        cfg['in_channels'],
        cfg['num_kernel_layers'],
        cfg['loss']
    )
    return merge_head
