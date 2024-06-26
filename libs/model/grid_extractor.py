import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align
from .utils import poly2bbox, poly2obb, obb2poly
from mmcv.ops import RoIAlignRotated


def convert_to_roi_format(lines_box):
    concat_boxes = torch.cat(lines_box, dim=0)
    device = concat_boxes.device
    ids = torch.cat(
        [
            torch.full((lines_box_pi.shape[0], 1), i, dtype=torch.float, device=device)
            for i, lines_box_pi in enumerate(lines_box)
        ],
        dim=0
    )
    rois = torch.cat([ids, concat_boxes], dim=1)
    return rois


def align_cells_feat(cells_feat, num_rows, num_cols):
    batch_size = len(cells_feat)
    dtype = cells_feat[0].dtype
    device = cells_feat[0].device

    max_row_nums = max(num_rows)
    max_col_nums = max(num_cols)

    aligned_cells_feat = list()
    masks = torch.zeros([batch_size, max_row_nums, max_col_nums], dtype=dtype, device=device)
    for batch_idx in range(batch_size):
        num_rows_pi = num_rows[batch_idx]
        num_cols_pi = num_cols[batch_idx]
        cells_feat_pi = cells_feat[batch_idx]
        cells_feat_pi = cells_feat_pi.transpose(0, 1).reshape(-1, num_rows_pi, num_cols_pi)
        aligned_cells_feat_pi = F.pad(
            cells_feat_pi,
            (0, max_col_nums-num_cols_pi, 0, max_row_nums-num_rows_pi, 0, 0),
            mode='constant',
            value=0
        )
        aligned_cells_feat.append(aligned_cells_feat_pi)

        masks[batch_idx, :num_rows_pi, :num_cols_pi] = 1
    aligned_cells_feat = torch.stack(aligned_cells_feat, dim=0)
    return aligned_cells_feat, masks


class RoiPosFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim

        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )

        self.bbox_ln = nn.LayerNorm(self.output_dim)
        self.bbox_tranform = nn.Linear(4, self.output_dim)

        self.add_ln = nn.LayerNorm(self.output_dim)

    def forward(self, feats, lines_box, img_sizes):
        lines_box = [item.float() for item in lines_box] # confirm the type is float for rois
        bboxes = [poly2bbox(item) for item in lines_box]
        rois = convert_to_roi_format(bboxes)

        lines_feat = roi_align(
            input=feats,
            boxes=rois,
            output_size=self.pool_size,
            spatial_scale=self.scale,
            sampling_ratio=2
        )
        
        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.proj(lines_feat)
        lines_feat = list(torch.split(lines_feat, [item.shape[0] for item in lines_box]))

        # Add Pos Embedding
        for idx, (line_box, img_size) in enumerate(zip(bboxes, img_sizes)):
            line_box[:, 0::2] = line_box[:, 0::2] / img_size[0]
            line_box[:, 1::2] = line_box[:, 1::2] / img_size[1]
            lines_feat[idx] = self.add_ln(lines_feat[idx] + self.bbox_ln(self.bbox_tranform(line_box)))

        return list(lines_feat)


class CornerPosFeatExtraxtor(nn.Module):
    def __init__(self, scale, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.output_dim = output_dim
        self.bbox_ln = nn.LayerNorm(self.output_dim)
        self.bbox_tranform = nn.Linear(2, self.output_dim)
        self.corner_add = nn.Conv2d(output_dim, output_dim, 2,2)
        self.corner_proj = nn.Sequential( 
                                         nn.Linear(input_dim, output_dim),
                                         )
        self.add_ln = nn.LayerNorm(self.output_dim)

    def forward(self, num_rows, num_cols, feat, boxes):
        corner_feats, corner_positions = self.smaple_corner( boxes, num_rows, num_cols, feat )
        corner_feats_position = list()
        for batch_index in range(feat.shape[0]):
            corner_feats_proj = self.corner_proj(corner_feats[batch_index].permute(0,2,3,1))
            corner_position_proj = self.bbox_ln(self.bbox_tranform( corner_positions[batch_index]))

            feat_position =  ( corner_feats_proj+ corner_position_proj ).permute(0,3,1,2)
            feat_position_fusion = self.corner_add(feat_position)
            feat_position_fusion = self.add_ln(feat_position_fusion.permute(0,2,3,1)).permute(0,3,1,2)
            # print(feat_position_fusion.shape)
            corner_feats_position.append(feat_position_fusion)
            
        corner_feat_align,corner_feat_align_mask = self.align_corner_feat( num_rows, num_cols, corner_feats_position )

        return corner_feat_align,corner_feat_align_mask

    def align_corner_feat(self, num_rows, num_cols,feat):
        batch_size = len(feat)
        dtype = feat[0].dtype
        device = feat[0].device

        max_row_nums = max(num_rows)
        max_col_nums = max(num_cols)

        aligned_feat = list()
        masks = torch.zeros([batch_size, max_row_nums, max_col_nums], dtype=dtype, device=device)
        for batch_idx in range(batch_size):
            num_rows_pi = num_rows[batch_idx]
            num_cols_pi = num_cols[batch_idx]
            feat_pi = feat[batch_idx]
            aligned_feat_pi = F.pad(
                feat_pi,
                (0, max_col_nums-num_cols_pi, 0, max_row_nums-num_rows_pi),
                mode='constant',
                value=0
            )
            # print(aligned_feat_pi.shape, feat_pi.shape)
            aligned_feat.append(aligned_feat_pi)

            masks[batch_idx, :num_rows_pi, :num_cols_pi] = 1
        aligned_feat = torch.cat(aligned_feat, dim=0)
        return aligned_feat, masks

    def smaple_corner(self, polys, num_rows, num_cols, feat):
        feat_h, feat_w = feat.shape[2:]
        corner_feats = list()
        corner_positions = list()
        for batch_index in range(feat.shape[0]):
            polys_batch = polys[batch_index].reshape((num_rows[batch_index], num_cols[batch_index],4,2))*self.scale
            polys_batch_permute = polys_batch.permute(3,2,0,1)
            polys_batch_grid = F.pixel_shuffle( polys_batch_permute, 2 ).squeeze(1)*self.scale
            polys_batch_grid_x = (polys_batch_grid[0]/feat_w-0.5)*2
            polys_batch_grid_y = (polys_batch_grid[1]/feat_h-0.5)*2
            grids = torch.stack([polys_batch_grid_x, polys_batch_grid_y], dim=-1).unsqueeze(0)

            corner_feat = F.grid_sample( feat[batch_index].unsqueeze(0), grids)
            corner_feats.append( corner_feat)
            corner_positions.append((grids))

            # print(corner_feat.shape,num_rows[batch_index], num_cols[batch_index] )
        
        return corner_feats, corner_positions
        

class RRoiPosFeatExtraxtor(nn.Module):
    def __init__(self, scale, pool_size, input_dim, output_dim):
        super().__init__()
        self.scale = scale
        self.pool_size = pool_size
        self.output_dim = output_dim

        self.pooler = RoIAlignRotated(
            out_size=pool_size,
            spatial_scale=scale,
            sample_num=2
        )

        input_dim = input_dim * self.pool_size[0] * self.pool_size[1]
        self.proj = nn.Sequential(
            nn.Linear(input_dim, self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim, self.output_dim)
        )
        
        self.bbox_ln = nn.LayerNorm(self.output_dim)
        self.bbox_tranform = nn.Linear(8, self.output_dim)

        self.add_ln = nn.LayerNorm(self.output_dim)

    def forward(self, feats, lines_box, img_sizes):
        lines_box = [item.float() for item in lines_box] # confirm the type is float for rois
        obbs = [poly2obb(item) for item in lines_box]
        polys = [obb2poly(item) for item in obbs]
        rois = convert_to_roi_format(obbs)
        lines_feat = self.pooler(feats, rois)

        lines_feat = lines_feat.reshape(lines_feat.shape[0], -1)
        lines_feat = self.proj(lines_feat)
        lines_feat = list(torch.split(lines_feat, [item.shape[0] for item in lines_box]))

        # Add Pos Embedding
        for idx, (line_box, img_size) in enumerate(zip(polys, img_sizes)):
            line_box[:, 0::2] = line_box[:, 0::2] / img_size[0] # divide by image_w
            line_box[:, 1::2] = line_box[:, 1::2] / img_size[1] # divide by image_h
            lines_feat[idx] = self.add_ln(lines_feat[idx] + self.bbox_ln(self.bbox_tranform(line_box)))

        return list(lines_feat)


class SALayer(nn.Module):
    '''
        A simplified version of Transformer
    '''
    def __init__(self, in_dim, att_dim, head_nums):
        super().__init__()
        self.in_dim = in_dim
        self.att_dim = att_dim
        self.head_nums = head_nums

        assert self.in_dim % self.head_nums == 0

        self.key_layer = nn.Conv1d(self.in_dim, self.att_dim , 1, 1, 0)
        self.query_layer = nn.Conv1d(self.in_dim, self.att_dim, 1, 1, 0)
        self.value_layer = nn.Conv1d(self.in_dim, self.in_dim, 1, 1, 0)
        self.scale = 1 / math.sqrt(self.att_dim)

    def forward(self, feats, masks=None):
        bs, c, n = feats.shape
        keys = self.key_layer(feats).reshape(bs, -1, self.head_nums, n)
        querys = self.query_layer(feats).reshape(bs, -1, self.head_nums, n)
        values = self.value_layer(feats).reshape(bs, -1, self.head_nums, n)

        logits = torch.einsum('bchk,bchq->bhkq', keys, querys) * self.scale
        if masks is not None:
            logits = logits - (1 - masks[:, None, :, None]) * 1e8
        weights = torch.softmax(logits, dim=2)

        new_feats = torch.einsum('bchk,bhkq->bchq', values, weights)
        new_feats = new_feats.reshape(bs, -1, n)
        return new_feats + feats


class GridExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, grid_type, pool_size, scale, \
        num_attention_layers, num_attention_heads, intermediate_size, dropout_prob):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_type = grid_type
        self.pool_size = pool_size
        self.scale = scale

        assert grid_type in ['bbox', 'obb', 'corner']
        if grid_type == 'bbox':
            self.box_feat_extractor = RoiPosFeatExtraxtor(
                self.scale,
                self.pool_size,
                self.in_channels,
                self.out_channels
            )
        elif grid_type == 'obb':
            self.box_feat_extractor = RRoiPosFeatExtraxtor(
                self.scale,
                self.pool_size,
                self.in_channels,
                self.out_channels
            )
        else:
            self.box_feat_extractor = RoiPosFeatExtraxtor(
                self.scale,
                self.pool_size,
                self.in_channels,
                self.out_channels
            )
            self.corner_extractor = CornerPosFeatExtraxtor( self.scale, self.in_channels, self.out_channels )

        self.num_attention_layers = num_attention_layers
        self.row_sas = nn.ModuleList()
        self.col_sas = nn.ModuleList()
        for _ in range(self.num_attention_layers):
            self.row_sas.append(SALayer(out_channels, out_channels, num_attention_heads))
            self.col_sas.append(SALayer(out_channels, out_channels, num_attention_heads))


    def forward(self, feats, num_rows, num_cols, bboxes, img_sizes):
        grid_feats = self.box_feat_extractor(feats, bboxes, img_sizes)
        aligned_cells_feat, masks = align_cells_feat(grid_feats, num_rows, num_cols)

        if self.grid_type=="corner":
            corner_feat, corner_mask = self.corner_extractor(num_rows, num_cols, feats, bboxes  )
            aligned_cells_feat = aligned_cells_feat+corner_feat

        bs, c, nr, nc = aligned_cells_feat.shape

        for idx in range(self.num_attention_layers):
            col_cells_feat = aligned_cells_feat.permute(0, 2, 1, 3).contiguous().reshape(bs * nr, c, nc)
            col_masks = masks.reshape(bs * nr, nc)
            col_cells_feat = self.col_sas[idx](col_cells_feat, col_masks) # self-attention
            aligned_cells_feat = col_cells_feat.reshape(bs, nr, c, nc).permute(0, 2, 1, 3).contiguous()

            row_cells_feat = aligned_cells_feat.permute(0, 3, 1, 2).contiguous().reshape(bs * nc, c, nr)
            row_masks = masks.transpose(1, 2).reshape(bs * nc, nr)
            row_cells_feat = self.row_sas[idx](row_cells_feat, row_masks) # self-attention
            aligned_cells_feat = row_cells_feat.reshape(bs, nc, c, nr).permute(0, 2, 3, 1).contiguous()


        return aligned_cells_feat, masks


def build_grid_extractor(cfg):
    grid_extractor = GridExtractor(
        in_channels=cfg['in_channels'],
        out_channels=cfg['out_channels'],
        grid_type=cfg['grid_type'],
        pool_size=cfg['pool_size'],
        scale=cfg['scale'],
        num_attention_layers=cfg['num_attention_layers'],
        num_attention_heads=cfg['num_attention_heads'],
        intermediate_size=cfg['intermediate_size'],
        dropout_prob=cfg['dropout_prob']
    )
    return grid_extractor