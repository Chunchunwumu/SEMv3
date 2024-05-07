import math
import torch
from torch import nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from collections import OrderedDict
from .resa_head import build_resa_head
import matplotlib.pyplot as plt
from libs.utils.metric import cal_segment_pr
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from typing import Callable, Optional, List


def cal_segments(cls_logits):
    cls_probs, cls_ids = torch.max(cls_logits, dim=1)
    cls_probs = cls_probs.tolist()
    cls_ids = cls_ids.tolist()

    segments = list()

    for idx in range(len(cls_ids)):
        if idx == 0:
            if cls_ids[idx] == 1:
                segments.append([idx, cls_probs[idx]])
            
        elif cls_ids[idx] == 1:
            
            if cls_ids[idx - 1] == 1:
                if cls_probs[idx] > segments[-1][1]:
                    segments[-1] = [idx, cls_probs[idx]]
            else:
                segments.append([idx, cls_probs[idx]])
    
    segments = [item[0] for item in segments]
    if len(segments) < 2:
        segments = [0, cls_logits.shape[0]-1] # é¦–å°¾
    return segments

def cal_segment_loss(cls_logits, det_bboxes, stride, line_type):
    device = cls_logits.device
    bg_cls_loss = list()
    fg_ctc_loss = list()
    for cls_logits_pi, bboxes in zip(cls_logits, det_bboxes):# every sample
        if line_type == 'row':
            _, start_ids, _, end_ids = torch.split(bboxes/stride, 1, dim=-1) # (N, 1)
        else:
            start_ids, _, end_ids, _ = torch.split(bboxes/stride, 1, dim=-1) # (N, 1)
        bg_target = torch.full([cls_logits_pi.shape[0]], 0, dtype=torch.long, device=device)
        for start_id, end_id in zip(start_ids, end_ids):
            bg_target[int(start_id):int(end_id+1)] = -1 # remove foreground pixel
        bg_cls_loss_pi = F.cross_entropy(cls_logits_pi, bg_target, ignore_index=-1)
        bg_cls_loss.append(bg_cls_loss_pi)

        if start_ids.size(0) > 0:
            fg_logits = [cls_logits_pi[int(start_id):int(end_id+1), :] for start_id, end_id in zip(start_ids, end_ids)]
            fg_logits_length = [item.shape[0] for item in fg_logits]
            fg_max_length = max(fg_logits_length)

            fg_logits = torch.stack([F.pad(item, (0, 0, 0, fg_max_length-item.shape[0])) for item in fg_logits], dim=1)
            fg_logits = torch.log_softmax(fg_logits, dim=2)
            fg_logits_length = torch.tensor(fg_logits_length, dtype=torch.long, device=device)

            fg_target = torch.ones([fg_logits_length.shape[0], 1], dtype=torch.long, device=device)# 每一个box 只有一个真值，并且真值为1. 使模型自己确定哪一个位置为1
            fg_target_length = torch.ones_like(fg_logits_length)

            fg_ctc_loss_pi = F.ctc_loss(fg_logits, fg_target, fg_logits_length, fg_target_length, zero_infinity=True)
            fg_ctc_loss.append(fg_ctc_loss_pi)

    bg_cls_loss = torch.mean(torch.stack(bg_cls_loss))
    fg_ctc_loss = torch.mean(torch.stack(fg_ctc_loss))
    return bg_cls_loss, fg_ctc_loss

class delta_feature_fusion( nn.Module ):
    def __init__(self, inchannels):
        super().__init__()
        self.in_channels = inchannels
        self.mean_feature_trans = nn.Linear(inchannels, inchannels)
        self.points_feature_trans = nn.Linear(inchannels, inchannels)
        self.fusion = nn.Linear(inchannels, inchannels)
    def forward(self, mean_feature, points_feature ):
        N,K,_ = points_feature.shape
        # print(mean_feature.shape, points_feature.shape,mean_feature[:,None,:].repeat((1,K,1)).shape)
        # mean feature:N*C 与 feature concate:N*K*C
        mean_feature_trans = self.mean_feature_trans(mean_feature[:,None,:].repeat((1,K,1)))
        points_feature_trans = self.points_feature_trans(points_feature)
        return self.fusion( mean_feature_trans + points_feature_trans )


class SplitHead_point_delta_conv_2(nn.Module):
    """
    非密集的采样kernel，每个instance 固定采样n 个kernel feature
    然后每个kernel 特征预测一个分割图，在分割图上做融合，但是每个分割图都需要做损失
    """

    def __init__(self, line_type, down_stride, resa, in_channels, loss, feature_branch_kernelSzie=1, kernel_branch_kernelSize=1, feature_branch_act=None):
        super().__init__()
        assert line_type in ['row', 'col']
        self.line_type = line_type

        # init down sample block --- Detection banch
        blocks = OrderedDict()
        down_kernel_size = (1, 2) if line_type =='row' else (2,1)
        for i in range(int(math.log2(down_stride))):
            name_prefix = 'downsample' + str(i + 1)
            blocks[name_prefix + '_maxpool'] = nn.MaxPool2d(down_kernel_size)
            blocks[name_prefix + '_conv'] = ConvModule(in_channels, in_channels, \
                kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU'))
        self.det_down_sample_blocks = nn.Sequential(blocks)
        self.det_resa_head = build_resa_head(resa)
        self.det_conv = ConvModule(
            in_channels,
            2, # background/foreground prob
            kernel_size=1,
            conv_cfg=None,
            act_cfg=None)
        # init down sample block --- Kernel banch
        blocks = OrderedDict()
        down_kernel_size = (1, 2) if line_type =='row' else (2,1)
        for i in range(int(math.log2(down_stride))):
            name_prefix = 'downsample' + str(i + 1)
            blocks[name_prefix + '_maxpool'] = nn.MaxPool2d(down_kernel_size)
            blocks[name_prefix + '_conv'] = ConvModule(in_channels, in_channels, \
                kernel_size=3, stride=1, padding=1, act_cfg=dict(type='ReLU'))
        self.down_sample_blocks = nn.Sequential(blocks)
        self.kernel_resa_head = build_resa_head(resa)
        self.kernel_conv = ConvModule(
            in_channels,
            in_channels, # in_channels + bias
            kernel_size=kernel_branch_kernelSize,
            padding=kernel_branch_kernelSize//2,
            conv_cfg=None,
            act_cfg=None)
        self.feats_conv = ConvModule(
            in_channels,
            in_channels,
            kernel_size=feature_branch_kernelSzie,
            padding=feature_branch_kernelSzie//2,
            conv_cfg=None,
            act_cfg=feature_branch_act)
        # init feature conv -- Feature branch
        self.delta_feature_fusion = delta_feature_fusion(in_channels)
        if self.line_type =="row":
            self.delta_predict = nn.Sequential(nn.Conv2d(in_channels, in_channels, (1,3),1,(0,1)),
                nn.ReLU(),
                nn.Conv2d(in_channels, 1,1))
        else:
            self.delta_predict = nn.Sequential(nn.Conv2d(in_channels, in_channels, (3,1),1,(1,0)),
                nn.ReLU(),
                nn.Conv2d(in_channels, 1,1))

        assert loss['type'] in ['bce', 'focal', 'dice']
        self.loss_type = loss['type']
        assert loss['div'] in ['pos', 'all']
        
        self.loss_div = loss['div']
        self.loss_factor = loss['factor']

    def forward(self, feats, pad_shape, image_shape, det_bboxes, delta_labels, epoch_item=0):
        det_feats_ras = self.det_resa_head(self.det_down_sample_blocks(feats))
        det_feats = self.det_conv(det_feats_ras) 
        kernels = self.kernel_conv(self.kernel_resa_head(self.down_sample_blocks(feats))) 
        stride_kernel_feat = ( pad_shape[-1]/kernels.shape[-1], pad_shape[-2]/kernels.shape[-2] )

        _, C, _, _ = feats.shape
        stride = pad_shape[-1] / feats.shape[-1] # the factor to resize
        insert_row_col = [] 

        h, w = feats.shape[-2:]
        total_correct=0
        total_pred=0
        total_label=0
        delta_logits = []
        delta_process_label = []
        if self.line_type == 'row':
            stride_s = pad_shape[-1] / kernels.shape[-1] # the factor to resize

            center_points = []
            det_feats = det_feats.mean(-1).transpose(1, 2).contiguous() # (B, H, 2)# 0通道是背景，1是前景也就是分割线区域
            # kernels = kernels.mean(-1) # (B, C+1, H)
            for batch_idx in range(det_feats.shape[0]):
                # parsing the kernel position
                det_feats_pb = det_feats[batch_idx][:int(image_shape[batch_idx][1]/stride+2)] # (valid_H, 2) # 只对部分区域进行计算
                insert_row_yc, insert_row_index = [], []
                if not self.training: # inference stage
                    yc = torch.tensor(cal_segments(det_feats_pb), device=kernels.device) # (N, 1)
                    center_points.append(yc*stride)
                    insert_row_col.append([torch.tensor(insert_row_yc).to(det_feats_pb.device), insert_row_index])
                else: # during training stage, select from max prob from foreground
                    yc_cal = torch.tensor(cal_segments(det_feats_pb), device=kernels.device) # (N, 1)
                    # center_points_cal.append(yc_cal*stride)
                    _, y1, _, y2 = torch.split(det_bboxes[batch_idx]/stride, 1, dim=-1) # (N, 1) # bbox是N*4，行列的起始box
                    correct_nums, segment_nums, span_nums = cal_segment_pr(yc_cal,y1,y2)
                    total_correct+=correct_nums
                    total_pred+=segment_nums
                    total_label+=span_nums
                    yc = (y1 + y2) / 2 # (N, 1)
                    cls_probs_pi = torch.softmax(det_feats_pb, dim=1)[:, 1] # 每一个像素是前景的概率。

                    for row_idx in range(yc.size(0)): # 取出真实的前景区域预测结果中，最大的概率值，所在的位置
                        height = y2[row_idx] - y1[row_idx] # h一定是>1的。
                        if height > 1 and y2[row_idx]+1 < det_feats_pb.shape[0]:
                            span_cls_probs = cls_probs_pi[int(y1[row_idx][0]):int(y2[row_idx][0]+1)]
                            yc[row_idx][0] = torch.argmax(span_cls_probs).item() + y1[row_idx][0] # yc变成最大概率值，所在的位置

                    yc = [int(yc_t[0].item()) for yc_t in yc]
                    yc = [min(max(yc_t, 0), h-1) for yc_t in yc]
                    center_points.append(torch.tensor(yc).to(det_feats_pb.device)*stride) 

                    # append dynamic disturbance for merge module
                    if epoch_item > 0:
                        bg_spans = [[y2[index], y1[index+1]] for index in range(len(y1)-1) if int(y1[index+1])-int(y2[index])>=2]
                        for span in bg_spans:
                            span_cls_probs = cls_probs_pi[int(span[0]): int(span[1])]
                            segment = torch.argmax(span_cls_probs).item() + int(span[0]) 
                            if cls_probs_pi[segment] > 0.5 and segment not in yc:
                                insert_row_yc.append(segment)
                        insert_row_yc.sort()
                    yc_merge = sorted(yc + insert_row_yc)
                    insert_row_index = [yc_merge.index(row_) for row_ in insert_row_yc]
                    insert_row_yc = torch.tensor(insert_row_yc).to(det_feats_pb.device) * stride
                    insert_row_col.append([insert_row_yc, insert_row_index])

                if self.training:
                    points_kernels, mean_kernels, labels_delta = self.get_kernels(kernels[batch_idx],yc, h0 = int(image_shape[batch_idx][1]/stride_kernel_feat[1]+1), w0 = int(1+image_shape[batch_idx][0]/stride_kernel_feat[0]), delta_labels=delta_labels[batch_idx],center_points_labels=det_bboxes[batch_idx], stride=stride,stride_s=stride_s) # (N, C+1, k)
                    
                else:
                    points_kernels, mean_kernels, labels_delta = self.get_kernels(kernels[batch_idx],yc, h0 = int(max(yc[-1]+1,image_shape[batch_idx][1]/stride_kernel_feat[1]+1)), w0 = int(1+image_shape[batch_idx][0]/stride_kernel_feat[0]), stride=stride,stride_s=stride_s) # (N, C+1,k)
                
                deltas = self.delta_predict(self.delta_feature_fusion( mean_kernels, points_kernels ).permute(2,0,1).unsqueeze(0)).squeeze(0).permute(1,2,0)
                delta_logits.append(deltas)
                delta_process_label.append(labels_delta)

        else:
            center_points = []
            det_feats = det_feats.mean(-2).transpose(1, 2).contiguous() # (B, W, 2)
            
            stride_s = pad_shape[-2] / kernels.shape[-2] # the factor to resize
           
            for batch_idx in range(det_feats.shape[0]):
                # parsing the kernel position 
                insert_col_xc, insert_col_index = [], []
                det_feats_pb = det_feats[batch_idx][:int(image_shape[batch_idx][0]/stride+2)] # (valid_W, 2)
                if not self.training: # inference stage
                    xc = torch.tensor(cal_segments(det_feats_pb), device=kernels.device) # (N, 1)
                    center_points.append(xc*stride)
                    insert_row_col.append([torch.tensor(insert_col_xc).to(det_feats_pb.device), insert_col_index])
                else:
                    xc_cal = torch.tensor(cal_segments(det_feats_pb), device=kernels.device) # (N, 1)

                    x1, _, x2, _ = torch.split(det_bboxes[batch_idx]/stride, 1, dim=-1) # (N, 1)
                    correct_nums, segment_nums, span_nums = cal_segment_pr(xc_cal,x1,x2)
                    total_correct+=correct_nums
                    total_pred+=segment_nums
                    total_label+=span_nums
                    xc = (x1 + x2) / 2 # (N, 1)
                    cls_probs_pi = torch.softmax(det_feats_pb, dim=1)[:, 1]
                    for col_idx in range(xc.size(0)):
                        width = x2[col_idx] - x1[col_idx]
                        if width > 1 and x2[col_idx] + 1 < det_feats_pb.shape[0]:
                            span_cls_probs = cls_probs_pi[int(x1[col_idx][0]):int(x2[col_idx][0]+1)]
                            xc[col_idx][0] = torch.argmax(span_cls_probs).item() + x1[col_idx][0]

                    xc = [int(xc_t[0].item()) for xc_t in xc]
                    xc = [min(max(xc_t, 0), w-1) for xc_t in xc]
                    center_points.append(torch.tensor(xc).to(det_feats_pb.device)*stride) 

                    # append dynamic disturbance for merge module
                    if epoch_item > 0:
                        bg_spans = [[x2[index], x1[index+1]] for index in range(len(x1)-1) if int(x1[index+1])-int(x2[index])>=2]
                        for span in bg_spans:
                            span_cls_probs = cls_probs_pi[int(span[0]): int(span[1])]
                            segment = torch.argmax(span_cls_probs).item() + int(span[0]) 
                            if cls_probs_pi[segment] > 0.5 and segment not in xc:
                                insert_col_xc.append(segment)
                        insert_col_xc.sort()
                    xc_merge = sorted(xc + insert_col_xc)
                    insert_col_index = [xc_merge.index(col_) for col_ in insert_col_xc]
                    insert_col_xc = torch.tensor(insert_col_xc).to(det_feats_pb.device) * stride
                    insert_row_col.append([insert_col_xc, insert_col_index])
                if self.training:
                    points_kernels, mean_kernels, labels_delta = self.get_kernels(kernels[batch_idx],xc, h0 = int(image_shape[batch_idx][1]/stride_kernel_feat[1]+1), w0 = int(1+image_shape[batch_idx][0]/stride_kernel_feat[0]), delta_labels=delta_labels[batch_idx],center_points_labels=det_bboxes[batch_idx], stride=stride,stride_s=stride_s) # (N, C+1,k)
                else:
                    points_kernels, mean_kernels, labels_delta = self.get_kernels(kernels[batch_idx],xc, h0 = int(image_shape[batch_idx][1]/stride_kernel_feat[1]+1), w0 = int(max(xc[-1]+1,1+image_shape[batch_idx][0]/stride_kernel_feat[0])), stride=stride,stride_s=stride_s) # (N, C+1,k)

                deltas = self.delta_predict(self.delta_feature_fusion( mean_kernels, points_kernels ).permute(2,1,0).unsqueeze(0)).squeeze(0).permute(2,1,0)
                delta_logits.append(deltas)
                delta_process_label.append(labels_delta)
        result_info = dict()
        
        # cal det p r 
        if self.training:
            if total_pred != 0:
                result_info['det_p'] = total_correct/total_pred
            else:
                result_info['det_p'] = 0

            if total_label != 0:
                result_info['det_r'] = total_correct/total_label
            else:
                result_info['det_r'] = 0

        # BCE and CTC loss for segment
        if self.training:
            bg_cls_loss, fg_ctc_loss = cal_segment_loss(det_feats, det_bboxes, stride, self.line_type) # det loss， not segment loss
            result_info['bg_cls_loss'] = bg_cls_loss
            result_info['fg_ctc_loss'] = fg_ctc_loss
        
        if self.training:
            delta_loss = []
            for batch_idx in range(det_feats.shape[0]):
                loss_batch = F.mse_loss( delta_logits[batch_idx][...,0], delta_process_label[batch_idx] )
                delta_loss.append(loss_batch)
            result_info["delta_loss"] = sum(delta_loss)/len(delta_loss)
        
        # return result_info, delta_process_label, center_points, insert_row_col
        if self.training:
            return result_info, delta_process_label, center_points, insert_row_col
        else:
            delta_pred = [i[...,0] for i in delta_logits]
            return result_info, delta_pred, center_points, insert_row_col

    def get_kernels(self,kernels, id_c0, h0, w0, center_points_labels=None, delta_labels=None,stride=4,stride_s=32 ):
        # 每次都取一整行/列的kernel feature
        # 取对应的delta labels
        # 对delta labels与feature的下采样尺度对齐
        # 利用center points label调整delta的偏移量
        
        h,w = kernels.shape[-2], kernels.shape[-1]
        h0=min(h0,h)
        w0=min(w0,w)
        if self.training:
            x1,y1,x2,y2 = torch.split(center_points_labels, 1, dim=-1)
            xc=(x1+x2)/2/stride
            yc=(y1+y2)/2/stride
        kernels_valid = kernels[:, :h0, :w0]
        labels_delta=None
        if self.line_type == "row":
            points_kernels = kernels_valid[:,id_c0, :].permute(1,2,0)
            mean_kernels = torch.mean(points_kernels, dim=1)
            if self.training:
                delta = torch.tensor(id_c0).to(yc)[:,None] - yc
                labels_delta = self.get_delta_labels(w0,delta_labels,stride_s).to(delta)
                labels_delta -= delta*stride
                labels_delta = labels_delta/15
        
        else:
            points_kernels = kernels_valid[:, :,id_c0].permute(2,1,0)
            mean_kernels = torch.mean(points_kernels, dim=1)
            if self.training:
                delta = torch.tensor(id_c0).to(xc)[:,None] - xc
                labels_delta = self.get_delta_labels(h0,delta_labels,stride_s).to(delta)
                labels_delta -= delta*stride
                labels_delta = labels_delta/15
        # print(labels_delta.shape, points_kernels.shape[0:2],h0)
        return points_kernels, mean_kernels, labels_delta

    def get_delta_labels(self, ids, labels,stride_s):
        # print(stride_s)
        stride_s = int(stride_s)
        N,K = labels.shape
        c_32 = torch.arange(ids)*stride_s
        width = torch.ceil((c_32[-1]-(K-1))/stride_s)

        if  width>0:
            label_1 = labels[:,c_32[:-int(width)]]
            label_2 = labels[:,-1:].repeat((1,int(width)))
            label_kernel = torch.cat( [label_1, label_2], dim=-1 )
        else:
            label_kernel = labels[:,c_32]
        return label_kernel




def build_split_head_1d_delta_conv_2(cfg):
    splitHead = SplitHead_point_delta_conv_2(
        cfg['line_type'],
        cfg['down_stride'],
        cfg['resa'],
        cfg['in_channels'],
        cfg['loss'],
        cfg['feature_branch_kernelSzie'],
        cfg['kernel_branch_kernelSize'],
        cfg['feature_branch_act']
    )
    return splitHead
