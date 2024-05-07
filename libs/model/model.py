import torch
from torch import nn
from mmdet.models import build_backbone
from .neck import build_neck_2
from .posemb import build_posemb_head
from .split_head_1d import build_split_head_1d_delta_conv_2
from .grid_extractor import build_grid_extractor
from .merge_head import build_merge_head_new_token_v1

from .utils import layout_update_new_token, parse_grid_bboxes_delta_np

class Model_points_delta_conv_2_new_merger_v1(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.neck = build_neck_2(cfg.neck)
        self.posemb = build_posemb_head(cfg.posemb) #加入了空间信息的图像特征。
        

        self.row_split_head = build_split_head_1d_delta_conv_2(cfg.row_split_head)
        self.col_split_head = build_split_head_1d_delta_conv_2(cfg.col_split_head)
        self.grid_extractor = build_grid_extractor(cfg.grid_extractor)
        self.merge_head = build_merge_head_new_token_v1(cfg.merge_head)
    
    def forward(self, images, images_size, row_start_bboxes=None,  \
        col_start_bboxes=None,  layouts=None, epoch_item=0,col_deltas=None,row_deltas=None):
        result_info = dict()
        feats = self.backbone(images)
        feats = self.neck(feats) # (B, C, H/4, W/4)
        feats = tuple([self.posemb(feats[0])]) # (B, C, H/4, W/4)
        
        # row table line instance segmentation
        rs_result_info, row_deltas, row_center_points, insert_rows = self.row_split_head(feats[0], images.shape, images_size, row_start_bboxes, row_deltas, epoch_item)
        rs_result_info = {'row_%s' % key: val for key, val in rs_result_info.items()}
        result_info.update(rs_result_info)

        # col table line instance segmentation
        cs_result_info, col_deltas, col_center_points, insert_cols = self.col_split_head(feats[0], images.shape, images_size, col_start_bboxes, col_deltas, epoch_item)
        cs_result_info = {'col_%s' % key: val for key, val in cs_result_info.items()}
        result_info.update(cs_result_info)

        # parse the table grid bboxes
        with torch.no_grad():
            stride_w = images.shape[3] / feats[0].shape[3]
            stride_h = images.shape[2] / feats[0].shape[2]
            
            table_gird_bboxes, num_rows, num_cols = parse_grid_bboxes_delta_np(row_center_points, row_deltas, \
                col_center_points, col_deltas, stride_w, stride_h, images.shape, insert_rows, insert_cols, score_threshold=0.25) # batch tensor -> [(N, 4)]

        # extract the grid-level features
        grid_feats, grid_masks = self.grid_extractor(feats[0], num_rows, num_cols, table_gird_bboxes, images_size)
        
        # grid merge predict
        if self.training:
            layouts = layout_update_new_token(layouts, insert_rows, insert_cols, row_center_points, col_center_points)
        
        mg_result_info, mg_logits = self.merge_head(grid_feats, grid_masks, layouts)
      
        mg_result_info = {'merge_%s' % key: val for key, val in mg_result_info.items()}
        result_info.update(mg_result_info)
        return result_info, row_start_bboxes, row_center_points, row_deltas, \
            col_start_bboxes, col_center_points, col_deltas, mg_logits, num_rows, num_cols, table_gird_bboxes
