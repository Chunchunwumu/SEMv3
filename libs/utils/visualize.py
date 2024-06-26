import cv2
import math
import copy
import random
import operator
import numpy as np
import seaborn as sns
from functools import reduce
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F


def process_layout(score, index, use_score=False, is_merge=True, score_threshold=0.5):
    if use_score:
        if is_merge:
            y, x = torch.where(score < score_threshold)
            index[y, x] = index.max() + 1
        else:
            y, x = torch.where(score < score_threshold)
            index[y, x] = torch.arange(index.max() + 1, index.max() + 1 + len(y)).to(index.device, index.dtype)

    layout = torch.full_like(index, -1)
    layout_mask = torch.full_like(index, -1)
    nrow, ncol = score.shape
    for cell_id in range(max(nrow * ncol, index.max() + 1)):
        if layout_mask.min() != -1:
            break
        crow, ccol = torch.where(layout_mask == layout_mask.min())
        ccol = ccol[crow == crow.min()].min()
        crow = crow.min()
        id = index[crow, ccol]
        h, w = torch.where(index == id)
        if h.shape[0] == 1 or w.shape[0] == 1: # single
            layout_mask[h, w] = 1
            layout[h, w] = cell_id
            continue
        else:
            h_min = h.min()
            h_max = h.max()
            w_min = w.min()
            w_max = w.max()
            if torch.all(index[h_min:h_max+1, w_min:w_max+1] == id):
                layout_mask[h_min:h_max+1, w_min:w_max+1] = 1
                layout[h_min:h_max+1, w_min:w_max+1] = cell_id
            else:
                lf_row = crow
                lf_col = ccol
                col_mem = -1
                for col_ in range(lf_col, w_max + 1):
                    if index[lf_row, col_] == id:
                        layout_mask[lf_row, col_] = 1
                        layout[lf_row, col_] = cell_id
                        col_mem = col_
                    else:
                        break
                for row_ in range(lf_row + 1, h_max + 1):
                    if torch.all(index[row_, lf_col: col_mem + 1] == id):
                        layout_mask[row_, lf_col: col_mem + 1] = 1
                        layout[row_, lf_col: col_mem + 1] = cell_id
                    else:
                        break
    return layout


def layout2spans(layout):
    rows, cols = layout.shape[-2:]
    cells_span = list()
    for cell_id in range(rows * cols):
        cell_positions = np.argwhere(layout == cell_id)
        if len(cell_positions) == 0:
            continue
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def trans2cellbbox(grid_bboxes):
    '''
    trans the input grid bboxes (N,4) to cell bbox
    '''
    grid_bboxes = np.array(grid_bboxes).reshape(-1, 2)
    x1 = int(grid_bboxes[:, 0].min())
    y1 = int(grid_bboxes[:, 1].min())
    x2 = int(grid_bboxes[:, 0].max())
    y2 = int(grid_bboxes[:, 1].max())
    return [x1, y1, x2, y2]


def trans2cellpoly(grid_polys):
    '''
    trans the input grid polys (N,8) to cell poly
    clock-wise
    '''
    grid_polys = np.array(grid_polys).reshape(-1, 2).tolist() # points
    grid_polys = [tuple(item) for item in grid_polys]
    grid_polys = list(set(grid_polys))
    # center = tuple(map(operator.truediv, reduce(lambda x, y: map(operator.add, x, y), grid_polys), [len(grid_polys)] * 2))
    # grid_polys = sorted(grid_polys, key=lambda coord: (-135 - math.degrees(math.atan2(*tuple(map(operator.sub, coord, center))[::-1]))) % 360, reverse=True)
    # grid_polys = [list(item) for item in grid_polys]
    grid_polys = sorted_points_clock_wise(grid_polys)
    return grid_polys

def trans2cellpoly_v1(grid_polys, position_row, position_col):

    '''
    trans the input grid polys (N,8) to cell poly
    clock-wise
    '''
    
    min_row = np.where(np.array(position_row)==min(position_row))[0].tolist()
    max_row = np.where(np.array(position_row)==max(position_row))[0].tolist()
    min_col = np.where(np.array(position_col)==min(position_col))[0].tolist()
    max_col = np.where(np.array(position_col)==max(position_col))[0].tolist()
    # print(min_row,min_col,max_row,max_col)
    grid_polys = np.array(grid_polys).reshape(-1,4, 2).tolist()
    # print(grid_polys)
    grid_points = list()
    # grid_points.append(  )
    up_points = list()
    for id, i in enumerate(min_row):
        if id ==0:
            up_points.append( grid_polys[i][0])
        up_points.append( grid_polys[i][1])
    grid_points.extend(up_points)
    right_points = list()
    for id, i in enumerate( max_col ):
        if id!=0:
            right_points.append( grid_polys[i][1] )
    grid_points.extend(right_points)

    down_points = list()
    for id, i in enumerate( max_row[::-1] ):
        if id==0:
            down_points.append( grid_polys[i][2] )
        down_points.append( grid_polys[i][3] )
    grid_points.extend(down_points)

    left_points = list()
    for id, i in enumerate( min_col[::-1]):
        if id!=0:
            left_points.append( grid_polys[i][3] )
    
    grid_points.extend(left_points)

    return grid_points




def sorted_points_clock_wise( grid_polys ):
    grid_polys = np.array(grid_polys)
    center = np.mean( grid_polys, axis=0 )
    sub_center = grid_polys - center
    angles = np.arctan2( sub_center[:,1], sub_center[:,0] )
    sorted_id = sorted( range(len(angles)), key= lambda x: angles[x] )
    grid_polys = grid_polys[sorted_id,:].tolist()
    return grid_polys







def parse_cells(layout, spans, grid_bboxes, grid_polys):
    ''' grid polys 中顶点的顺序是 lt,lb,rb,rt '''
    grid_polys_change = grid_polys.copy()
    grid_polys_change[:,2:4] = grid_polys[:,6:8]
    grid_polys_change[:,6:8] = grid_polys[:,2:4]

    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        # print(cell_positions)
        # print(layout.shape)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys_change[cell_positions[:, 0]]
        
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        # print(row_indexs)
        # print(col_indexs)
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3])
        )
        cells.append(cell)

    return cells

def parse_cells_v2(layout, spans, grid_bboxes, grid_polys):

    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys[cell_positions[:, 0]]
        
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        # print(row_indexs)
        # print(col_indexs)
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3])
        )
        cells.append(cell)

    return cells


def parse_layout(mg_logits, num_rows, num_cols, score_threshold=0.5):
    num_grids = int(num_rows) * int(num_cols)
    mg_probs = mg_logits[:num_grids, :int(num_rows), :int(num_cols)].sigmoid() # (N, H, W)
    _, indices = (mg_probs > score_threshold).float().max(dim=0) # (H, W)
    values, _ = mg_probs.max(dim=0) # (H, W)
    layout = process_layout(values, indices, use_score=True, is_merge=False, score_threshold=score_threshold)
    layout = process_layout(values, layout)
    layout = layout.cpu().numpy()
    spans = layout2spans(layout)
    return layout, spans


def parse_grid_bboxes(row_center_points, row_segm_logits, col_center_points, col_segm_logits,\
    stride_w, stride_h, score_threshold=0.5, kernel_size=3, radius=1000):
    '''
        parse the start bboxes and segmentation results to table grid bboxes
    '''

    batch_grid_bboxes = []
    batch_num_rows = []
    batch_num_cols = []
    batch_col_segm_map = []
    batch_row_segm_map = []
    for batch_idx in range(len(row_center_points)):
        # parse start point
        rs_yc = row_center_points[batch_idx] # (NumRow, 1)
        cs_xc = col_center_points[batch_idx]# (NumCol, 1)
        rs_yc, rs_sorted_idx = torch.sort(rs_yc, descending=False) # sort (NumRow, 1)
        row_segm_logits_pb = row_segm_logits[batch_idx][rs_sorted_idx]# sort (NumRow, H, W)
        cs_xc, cs_sorted_idx = torch.sort(cs_xc, descending=False) # sort (NumCol, 1)
        col_segm_logits_pb = col_segm_logits[batch_idx][cs_sorted_idx] # sort (NumCol, H, W)

        # parse col line segmentation
        _, col_line_index = col_segm_logits_pb.max(dim=2) # (NumCol, H), (NumCol, H)
        col_segm_map = torch.zeros_like(col_segm_logits_pb) # (NumCol, H, W)
        col_segm_map = col_segm_map.scatter(2, col_line_index[:, :, None].expand_as(col_segm_map), 1.) # (NumCol, H, W)
        col_segm_map[col_segm_logits_pb.sigmoid() <= score_threshold] = 0. # remove background
        col_segm_map = F.max_pool2d(col_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur
        batch_col_segm_map.append(col_segm_map)

        # parse row line segmentation
        _, row_line_index = row_segm_logits_pb.max(dim=1) # (NumRow, W), (NumRow, W)
        row_segm_map = torch.zeros_like(row_segm_logits_pb) # (NumRow, H, W)
        row_segm_map = row_segm_map.scatter(1, row_line_index[:, None, :].expand_as(row_segm_map), 1.) # (NumRow, H, W)
        row_segm_map[row_segm_logits_pb.sigmoid() <= score_threshold] = 0. # remove background
        row_segm_map = F.max_pool2d(row_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur
        batch_row_segm_map.append(row_segm_map)

        # parse the poly bbox
        num_rows = rs_yc.size(0)
        num_cols = cs_xc.size(0)
        grid_polys = list()
        for row_idx in range(num_rows-1):
            for col_idx in range(num_cols-1):
                x1_segm_map = col_segm_map[col_idx] # (H, W)
                y1_segm_map = row_segm_map[row_idx] # (H, W)
                x2_segm_map = col_segm_map[col_idx+1] # (H, W)
                y2_segm_map = row_segm_map[row_idx+1] # (H, W)

                # left top coordinate
                lt_segm_map = x1_segm_map + y1_segm_map # (H, W)
                # valid_x1 = max(0, int(cs_xc[col_idx] // stride_w - radius))
                # valid_y1 = max(0, int(rs_yc[row_idx] // stride_h - radius))
                # valid_x2 = int(cs_xc[col_idx] // stride_w + radius)
                # valid_y2 = int(rs_yc[row_idx] // stride_h + radius)
                # valid_mask = torch.zeros_like(lt_segm_map)
                # valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                # lt_segm_map = lt_segm_map * valid_mask
                y_lt, x_lt = torch.where(lt_segm_map==2)
                if len(y_lt) > 0 and len(x_lt) > 0:
                    x_lt = int(x_lt.float().mean() * stride_w)
                    y_lt = int(y_lt.float().mean() * stride_h)
                else:
                    x_lt = int(cs_xc[col_idx])
                    y_lt = int(rs_yc[row_idx])

                # right top coordinate
                rt_segm_map = x2_segm_map + y1_segm_map # (H, W)
                # valid_x1 = max(0, int(cs_xc[col_idx+1] // stride_w - radius))
                # valid_y1 = max(0, int(rs_yc[row_idx] // stride_h - radius))
                # valid_x2 = int(cs_xc[col_idx+1] // stride_w + radius)
                # valid_y2 = int(rs_yc[row_idx] // stride_h + radius)
                # valid_mask = torch.zeros_like(rt_segm_map)
                # valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                # rt_segm_map = rt_segm_map * valid_mask
                y_rt, x_rt = torch.where(rt_segm_map==2)
                if len(y_rt) > 0 and len(x_rt) > 0:
                    x_rt = int(x_rt.float().mean() * stride_w)
                    y_rt = int(y_rt.float().mean() * stride_h)
                else:
                    x_rt = int(cs_xc[col_idx+1])
                    y_rt = int(rs_yc[row_idx])

                # right bottom coordinate
                rb_segm_map = x2_segm_map + y2_segm_map # (H, W)
                # valid_x1 = max(0, int(cs_xc[col_idx+1] // stride_w - radius))
                # valid_y1 = max(0, int(rs_yc[row_idx+1] // stride_h - radius))
                # valid_x2 = int(cs_xc[col_idx+1] // stride_w + radius)
                # valid_y2 = int(rs_yc[row_idx+1] // stride_h + radius)
                # valid_mask = torch.zeros_like(rb_segm_map)
                # valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                # rb_segm_map = rb_segm_map * valid_mask
                y_rb, x_rb = torch.where(rb_segm_map==2)
                if len(y_rb) > 0 and len(x_rb) > 0:
                    x_rb = int(x_rb.float().mean() * stride_w)
                    y_rb = int(y_rb.float().mean() * stride_h)
                else:
                    x_rb = int(cs_xc[col_idx+1])
                    y_rb = int(rs_yc[row_idx+1])

                # left bottom coordinate
                lb_segm_map = x1_segm_map + y2_segm_map # (H, W)
                # valid_x1 = max(0, int(cs_xc[col_idx] // stride_w - radius))
                # valid_y1 = max(0, int(rs_yc[row_idx+1] // stride_h - radius))
                # valid_x2 = int(cs_xc[col_idx] // stride_w + radius)
                # valid_y2 = int(rs_yc[row_idx+1] // stride_h + radius)
                # valid_mask = torch.zeros_like(lb_segm_map)
                # valid_mask[valid_y1:valid_y2, valid_x1:valid_x2] = 1.
                # lb_segm_map = lb_segm_map * valid_mask
                y_lb, x_lb = torch.where(lb_segm_map==2)
                if len(y_lb) > 0 and len(x_lb) > 0:
                    x_lb = int(x_lb.float().mean() * stride_w)
                    y_lb = int(y_lb.float().mean() * stride_h)
                else:
                    x_lb = int(cs_xc[col_idx])
                    y_lb = int(rs_yc[row_idx+1])

                grid_polys.append([x_lt, y_lt, x_rt, y_rt, x_rb, y_rb, x_lb, y_lb])

        if len(grid_polys) == 0:
            grid_polys.append([0, 0, 0, 0, 0, 0, 0, 0])
            num_cols = 2
            num_rows = 2

        grid_polys = torch.tensor(grid_polys, dtype=torch.float, device=row_segm_logits[0].device)
        # grid_polys[:, 0::2] *= stride_w
        # grid_polys[:, 1::2] *= stride_h

        batch_grid_bboxes.append(grid_polys)
        batch_num_cols.append(num_cols-1)
        batch_num_rows.append(num_rows-1)
    return batch_grid_bboxes, batch_num_rows, batch_num_cols, batch_row_segm_map, batch_col_segm_map


def parse_line_segm(row_center_points, row_segm_logits, col_center_points, col_segm_logits,\
    score_threshold=0.1, kernel_size=4, radius=1000):
    '''
        parse the start bboxes and segmentation results to table grid bboxes
    '''
    # parse start point
    rs_yc = row_center_points # (NumRow, 1)
    cs_xc = col_center_points # (NumCol, 1)
    rs_yc, rs_sorted_idx = torch.sort(rs_yc, descending=False) # sort (NumRow, 1)
    row_segm_logits = row_segm_logits[rs_sorted_idx] # sort (NumRow, H, W)
    cs_xc, cs_sorted_idx = torch.sort(cs_xc, descending=False) # sort (NumCol, 1)
    col_segm_logits = col_segm_logits[cs_sorted_idx] # sort (NumCol, H, W)

    # parse col line segmentation
    _, col_line_index = col_segm_logits.max(dim=2) # (NumCol, H), (NumCol, H)
    col_segm_map = torch.zeros_like(col_segm_logits) # (NumCol, H, W)
    col_segm_map = col_segm_map.scatter(2, col_line_index[:, :, None].expand_as(col_segm_map), 1.) # (NumCol, H, W)
    col_segm_map[col_segm_logits.sigmoid() <= score_threshold] = 0. # remove background
    col_segm_map = F.max_pool2d(col_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur

    # parse row line segmentation
    _, row_line_index = row_segm_logits.max(dim=1) # (NumRow, W), (NumRow, W)
    row_segm_map = torch.zeros_like(row_segm_logits) # (NumRow, H, W)
    row_segm_map = row_segm_map.scatter(1, row_line_index[:, None, :].expand_as(row_segm_map), 1.) # (NumRow, H, W)
    row_segm_map[row_segm_logits.sigmoid() <= score_threshold] = 0. # remove background
    row_segm_map = F.max_pool2d(row_segm_map, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/ 2)) # blur
    
    return row_segm_map, col_segm_map


# def draw_segm_logits(image, segm_logits, type, score_threshold=0.1, color='Reds'):
#     color_list = sns.color_palette("Reds", n_colors=255)
#     segm_logits = F.interpolate(segm_logits.unsqueeze(1), size=image.shape[:2]).squeeze(1)
#     segm_probs = segm_logits.sigmoid() # (N, H, W)
#     if type == 'row':
#         segm_softmax_prob = segm_probs.softmax(dim=1)
#     else: # type = 'col'
#         segm_softmax_prob = segm_probs.softmax(dim=2)
#     ratio = 0.9 / segm_softmax_prob.max()
#     segm_indexs = (segm_softmax_prob * len(color_list) * ratio).to(dtype=torch.int) # (N, H, W)
#     segm_indexs[segm_probs < score_threshold] = 0
#     image = torch.tensor(image).to(device=segm_logits.device) # (H, W, 3)
#     images = image.unsqueeze(0).repeat(segm_logits.size(0), 1, 1, 1) # (N, H, W, 3)
#     canvas = image.unsqueeze(0).repeat(segm_logits.size(0), 1, 1, 1)
#     for index in range(int(score_threshold*len(color_list)), len(color_list)):
#         segm_masks = segm_indexs == index
#         if segm_masks.sum() == 0: # no available pixel
#             continue
#         canvas[segm_masks] = torch.tensor([int(255*item) for item in color_list[index][::-1]], device=canvas.device, dtype=canvas.dtype)
#     images = images.float()/2 + canvas.float()/2
#     images = images.cpu().numpy().astype('uint8') # (N, H, W, 3)
#     return images


def draw_segm_logits(image, segm_logits, score_threshold=0.1, color='Reds'):
    color_list = sns.color_palette("Reds", n_colors=255)
    segm_logits = F.interpolate(segm_logits.unsqueeze(1), size=image.shape[:2]).squeeze(1)
    segm_probs = segm_logits.sigmoid() # (N, H, W)
    segm_indexs = (segm_probs * len(color_list)).to(dtype=torch.int) # (N, H, W)
    image = torch.tensor(image).to(device=segm_logits.device) # (H, W, 3)
    images = image.unsqueeze(0).repeat(segm_logits.size(0), 1, 1, 1) # (N, H, W, 3)
    canvas = image.unsqueeze(0).repeat(segm_logits.size(0), 1, 1, 1)
    for index in range(int(score_threshold*len(color_list)), len(color_list)):
        segm_masks = segm_indexs == index
        if segm_masks.sum() == 0: # no available pixel
            continue
        canvas[segm_masks] = torch.tensor([int(255*item) for item in color_list[index][::-1]], device=canvas.device, dtype=canvas.dtype)
    images = images.float()/2 + canvas.float()/2
    images = images.cpu().numpy().astype('uint8') # (N, H, W, 3)
    return images


def draw_col_wise(image, line_mask, line_color=(255,0,0), point_color=(0,255,0), interval=30, radius=3):
    H, W = image.shape[:2]
    for x_pos in range(0, W-1, interval):
        cv2.line(image, (x_pos, 0), (x_pos, H), line_color, 1)
        y_pos = torch.where(line_mask[:, x_pos] == 1)[0]
        if len(y_pos) == 0:
            continue
        y_pos = y_pos.float().mean()
        cv2.circle(image, (x_pos, int(y_pos)), radius, point_color, -1)
    return image


def draw_row_wise(image, line_mask, line_color=(255,0,0), point_color=(0,255,0), interval=30, radius=3):
    H, W = image.shape[:2]
    for y_pos in range(0, H-1, interval):
        cv2.line(image, (0, y_pos), (W-1, y_pos), line_color, 2)
        x_pos = torch.where(line_mask[y_pos, :] == 1)[0]
        if len(x_pos) == 0:
            continue
        x_pos = x_pos.float().mean()
        cv2.circle(image, (int(x_pos), y_pos), radius, point_color, -1)
    return image


def visualize(table, pred_table, pred_result, prefix, stride=4, back_color=(255,255,255), text_color=(255,0,0),  
    font_size=10, font_path='/train20/intern/permanent/zrzhang6/DocumentPretrain/dataprocess/process_rvlcdip/libs/simfang.ttf'):

    row_center_points, row_segm_logits, \
        col_center_points, col_segm_logits, \
            mg_logits, num_rows, num_cols, grid_polys = pred_result

    # # draw grid bboxes
    # grid_bboxes = grid_polys.reshape(-1, 4, 2) # (N, 4, 2)
    # x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    # y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    # x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    # y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    # grid_bboxes = [torch.stack([x1, y1, x2, y2], dim=-1)] # (N, 4)
   
    # # parse table layout
    # layout, spans = parse_layout(mg_logits, num_rows, num_cols)
    # cells = parse_cells(layout, spans, grid_bboxes[0].cpu().numpy(), grid_polys.cpu().numpy())
    
    # cells = pred_table['cells']

    # # # draw cell bbox
    # # image = copy.deepcopy(table['img'])
    # # for cell in cells:
    # #     x1, y1, x2, y2 = [int(item) for item in cell['bbox']]
    # #     color = (random.randint(0, 255), random.randint(0, 255),
    # #         random.randint(0, 255))
    # #     cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    # # # draw layout info
    # # image = Image.fromarray(image)
    # # draw_img = ImageDraw.Draw(image)
    # # for cell in cells:
    # #     x1, y1, *_ = [int(item) for item in cell['bbox']]
    # #     txt = '%d,%d,%d,%d' % (cell['col_start_idx'], cell['row_start_idx'], \
    # #         cell['col_end_idx'], cell['row_end_idx'])
    # #     num_txt = len(txt) - 3
    # #     # draw text backgroud
    # #     x2 = x1 + num_txt * font_size
    # #     y2 = y1 + font_size
    # #     draw_img.polygon([x1,y1,x2,y1,x2,y2,x1,y2], fill=back_color)
    # #     # draw text foreground
    # #     font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    # #     draw_img.text([x1, y1], txt, fill=text_color, font=font)
    # # cv2.imwrite(prefix+'_cell_bbox.png', cv2.resize(np.array(image),(image.size[0]*2, image.size[1]*2)))
    # # # cv2.imwrite(prefix+'_cell_bbox.png', cv2.resize(image, (image.shape[1], image.shape[0])))

    # # draw cell poly with alpha
    # image = copy.deepcopy(table['img'])
    # canvas = copy.deepcopy(table['img'])
    # for cell in cells:
    #     color = (255, 0, 0)
    #     poly = np.array(cell['segmentation']).reshape(-1,1,2).astype('int')
    #     cv2.polylines(canvas, [poly], True, color, 2)
    # image = (image/3 + canvas/3*2).astype(np.uint8)
    # cv2.imwrite(prefix+'_cell_poly_1.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))

    # # draw line in cell polys
    # for cell in cells:
    #     color = (random.randint(0, 255), random.randint(0, 255),
    #         random.randint(0, 255))
    #     line_ids = [int(idx) for idx in cell['transcript'].split('-') if len(idx) > 0]
    #     for idx in line_ids:
    #         poly = table['line_polys'][idx]
    #         poly = np.array(poly).reshape(-1, 1, 2).astype('int')
    #         cv2.polylines(image, [poly], True, color, 2)
    # cv2.imwrite(prefix+'_cell_poly_2.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))

    image = copy.deepcopy(table['img'])
    canvas = copy.deepcopy(table['img'])
    for cell in table['cells']:
        color = (255, 0, 0)
        poly = np.array(cell['segmentation']).reshape(-1,1,2).astype('int')
        cv2.polylines(canvas, [poly], True, color, 2)
    image = (image/3 + canvas/3*2).astype(np.uint8)
    cv2.imwrite(prefix+'_cell_poly_3.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))

    for cell in table['cells']:
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        line_ids = [int(idx) for idx in cell['transcript'].split('-') if len(idx) > 0]
        for idx in line_ids:
            poly = table['line_polys'][idx]
            poly = np.array(poly).reshape(-1, 1, 2).astype('int')
            cv2.polylines(image, [poly], True, color, 2)
    cv2.imwrite(prefix+'_cell_poly_4.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))

    # for poly in table['line_polys']:
    #     color = (random.randint(0, 255), random.randint(0, 255),
    #         random.randint(0, 255))
    #     poly = np.array(poly).reshape(-1, 1, 2).astype('int')
    #     cv2.polylines(image, [poly], True, color, 2)
    # cv2.imwrite(prefix+'_cell_poly_2.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))

    # draw cell poly
    # image = copy.deepcopy(table['img'])
    # for cell in cells:
    #     color = (random.randint(0, 255), random.randint(0, 255),
    #         random.randint(0, 255))
    #     poly = np.array(cell['segmentation']).reshape(-1,1,2).astype('int')
    #     cv2.polylines(image, [poly], True, color, 2)
    # cv2.imwrite(prefix+'_cell_poly_1.png', cv2.resize(image,(image.shape[1]*2, image.shape[0]*2)))
    # draw layout info
    # image = Image.fromarray(image)
    # draw_img = ImageDraw.Draw(image)
    # for cell in cells:
    #     x1, y1, *_ = [int(item) for item in cell['bbox']]
    #     txt = '%d,%d,%d,%d' % (cell['col_start_idx'], cell['row_start_idx'], \
    #         cell['col_end_idx'], cell['row_end_idx'])
    #     num_txt = len(txt) - 3
    #     # draw text backgroud
    #     x2 = x1 + num_txt * font_size
    #     y2 = y1 + font_size
    #     draw_img.polygon([x1,y1,x2,y1,x2,y2,x1,y2], fill=back_color)
    #     # draw text foreground
    #     font = ImageFont.truetype(font_path, font_size, encoding="utf-8")
    #     draw_img.text([x1, y1], txt, fill=text_color, font=font)
    # cv2.imwrite(prefix+'_cell_poly_2.png', cv2.resize(np.array(image),(image.size[0]*2, image.size[1]*2)))
    # cv2.imwrite(prefix+'_cell_poly.png', cv2.resize(image, (image.shape[1]*4, image.shape[0]*4)))
    
    # parse table line segment
    image = copy.deepcopy(table['img'])
    valid_h, valid_w = image.shape[:2]
    row_segm_logits = F.interpolate(row_segm_logits.unsqueeze(1), scale_factor=4).squeeze(1) # (N, H, W)
    col_segm_logits = F.interpolate(col_segm_logits.unsqueeze(1), scale_factor=4).squeeze(1) # (N, H, W)
    row_line_masks, col_line_masks = parse_line_segm(row_center_points, row_segm_logits, col_center_points, col_segm_logits, score_threshold=0.1, kernel_size=1)
    # row_line_masks = F.interpolate(row_line_masks.unsqueeze(1), size=image.shape[:2]).squeeze(1) # (N, H, W)
    # col_line_masks = F.interpolate(col_line_masks.unsqueeze(1), size=image.shape[:2]).squeeze(1) # (N, H, W)

    # draw row line
    image = copy.deepcopy(table['img'])
    canvas = copy.deepcopy(table['img'])
    for idx, mask in enumerate(row_line_masks):
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        yc, xc = torch.where(mask==1)
        xc, sorted_idx = torch.sort(xc, descending=False)
        yc = yc[sorted_idx]
        xc = xc.cpu().numpy()
        yc = yc.cpu().numpy()
        poly = list()
        for idx in range(0, len(yc)):
            poly.extend([int(xc[idx]), int(yc[idx])])
        poly = np.array(poly).reshape(-1, 1, 2).astype('int')
        cv2.polylines(canvas, [poly], False, color, 4)
    image = image/3 + canvas/3*2
    cv2.imwrite(prefix+'_row_line_segm.png', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))

    # draw column line
    image = copy.deepcopy(table['img'])
    canvas = copy.deepcopy(table['img'])
    for idx, mask in enumerate(col_line_masks):
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        yc, xc = torch.where(mask==1)
        yc, sorted_idx = torch.sort(yc, descending=False)
        xc = xc[sorted_idx]
        xc = xc.cpu().numpy()
        yc = yc.cpu().numpy()
        poly = list()
        for idx in range(0, len(yc), 8):
            poly.extend([int(xc[idx]), int(yc[idx])])
        poly = np.array(poly).reshape(-1, 1, 2).astype('int')
        cv2.polylines(image, [poly], False, color, 4)
    image = image/3 + canvas/3*2
    cv2.imwrite(prefix+'_col_line_segm.png', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))

    image = copy.deepcopy(table['img'])
    row_line_masks, col_line_masks = parse_line_segm(row_center_points, row_segm_logits, col_center_points, col_segm_logits, kernel_size=3)

    # draw row line segmentation
    row_line_masks = (row_line_masks > 0).cpu().numpy()
    image = copy.deepcopy(table['img'])
    canvas = copy.deepcopy(table['img'])
    for idx, mask in enumerate(row_line_masks):
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        canvas[mask[:valid_h, :valid_w]] = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
    image = image/3 + canvas/3*2
    cv2.imwrite(prefix+'_row_segm.png', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))

    # draw col line segmentation
    col_line_masks = (col_line_masks > 0).cpu().numpy()
    image = copy.deepcopy(table['img'])
    canvas = copy.deepcopy(table['img'])
    for idx, mask in enumerate(col_line_masks):
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        canvas[mask[:valid_h, :valid_w]] = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
    image = image/3 + canvas/3*2
    cv2.imwrite(prefix+'_col_segm.png', cv2.resize(image, (image.shape[1]*2, image.shape[0]*2)))

    # draw row segm logits
    # images = draw_segm_logits(copy.deepcopy(image), row_segm_logits, color='Reds') # (N, H, W, 3)
    # for idx, (row_image, row_line_mask) in enumerate(zip(images, row_line_masks)):
    #     # draw row line
    #     row_image = draw_col_wise(row_image, row_line_mask, interval=30, radius=5)
    #     cv2.imwrite(prefix+'_row_line_segm_%03d.png' % idx , row_image)
    #     row_line_image = copy.deepcopy(table['img'])
    #     line_mask = F.max_pool2d(row_line_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    #     row_line_image[(line_mask==1).cpu().numpy()] = np.array([255,0,0])
    #     cv2.imwrite(prefix+'_row_line_%03d.png' % idx , row_line_image)

    # #draw col segm logits
    # images = draw_segm_logits(copy.deepcopy(image), col_segm_logits, color='Reds') # (N, H, W, 3)
    # for idx, (col_image, col_line_mask) in enumerate(zip(images, col_line_masks)):
    #     # draw column line
    #     col_image = draw_row_wise(col_image, col_line_mask, interval=30, radius=5)
    #     cv2.imwrite(prefix+'_col_line_segm_%03d.png' % idx , col_image)
    #     col_line_image = copy.deepcopy(table['img'])
    #     line_mask = F.max_pool2d(col_line_mask.unsqueeze(0), kernel_size=5, stride=1, padding=2).squeeze(0)
    #     col_line_image[(line_mask==1).cpu().numpy()] = np.array([255,0,0])
    #     cv2.imwrite(prefix+'_col_line_%03d.png' % idx , col_line_image)

    # parse table line segment
    # row_line_masks, col_line_masks = parse_line_segm(row_center_points, row_segm_logits, col_center_points, col_segm_logits)

    # # draw row line segmentation
    # row_line_masks = F.interpolate(row_line_masks.unsqueeze(1).float(), size=image.shape[:2]).squeeze(1)
    # row_line_masks = (row_line_masks > 0).cpu().numpy()
    # all_image = copy.deepcopy(table['img'])
    # all_canvas = copy.deepcopy(table['img'])
    # for idx, mask in enumerate(row_line_masks):
    #     image = copy.deepcopy(table['img'])
    #     canvas = copy.deepcopy(table['img'])
    #     canvas[mask] = (0, 255, 0)
    #     image = image/2 + canvas/2
    #     cv2.imwrite(prefix+'_row_line_segm_%03d.png' % idx , cv2.resize(image, (image.shape[1]*4, image.shape[0]*4)))

    #     all_canvas[mask] = (random.randint(0, 255), random.randint(0, 255),
    #         random.randint(0, 255))
    # all_image = all_image/2 + all_canvas/2
    # cv2.imwrite(prefix+'_row_line_segm.png', cv2.resize(all_image, (all_image.shape[1]*4, all_image.shape[0]*4)))

    # # draw col line segmentation
    # col_line_masks = F.interpolate(col_line_masks.unsqueeze(1).float(), size=image.shape[:2]).squeeze(1)
    # col_line_masks = (col_line_masks > 0).cpu().numpy()
    # all_image = copy.deepcopy(table['img'])
    # all_canvas = copy.deepcopy(table['img'])
    # for idx, mask in enumerate(col_line_masks):
    #     image = copy.deepcopy(table['img'])
    #     canvas = copy.deepcopy(table['img'])
    #     canvas[mask] = (0, 255, 0)
    #     image = image/2 + canvas/2
    #     cv2.imwrite(prefix+'_col_line_segm_%03d.png' % idx , cv2.resize(image, (image.shape[1]*4, image.shape[0]*4)))

    #     all_canvas[mask] = (random.randint(0, 255), random.randint(0, 255),
    #         random.randint(0, 255))
    # all_image = all_image/2 + all_canvas/2
    # cv2.imwrite(prefix+'_col_line_segm.png', cv2.resize(all_image, (all_image.shape[1]*4, all_image.shape[0]*4)))   

    # # draw table row lines by model
    # row_segm_logits = F.interpolate(row_segm_logits.unsqueeze(1).float(), size=image.shape[:2]).squeeze(1)
    # row_segm_masks = (row_segm_logits.sigmoid() > 0.1).cpu().numpy()
    # all_image = copy.deepcopy(table['img'])
    # all_canvas = copy.deepcopy(table['img'])
    # for idx, mask in enumerate(row_segm_masks):
    #     image = copy.deepcopy(table['img'])
    #     canvas = copy.deepcopy(table['img'])
    #     canvas[mask] = (0, 255, 0)
    #     image = image/2 + canvas/2
    #     cv2.imwrite(prefix+'_row_line_segm_%03d.png' % idx , cv2.resize(image, (image.shape[1]*4, image.shape[0]*4)))

    #     all_canvas[mask] = (0, 255, 0)
    # all_image = all_image/2 + all_canvas/2
    # cv2.imwrite(prefix+'_row_line_segm.png', cv2.resize(all_image, (all_image.shape[1]*4, all_image.shape[0]*4)))

    # # draw table col lines
    # col_segm_logits = F.interpolate(col_segm_logits.unsqueeze(1).float(), size=image.shape[:2]).squeeze(1)
    # col_segm_masks = (col_segm_logits.sigmoid() > 0.5).cpu().numpy()
    # all_image = copy.deepcopy(table['img'])
    # all_canvas = copy.deepcopy(table['img'])
    # for idx, mask in enumerate(col_segm_masks):
    #     image = copy.deepcopy(table['img'])
    #     canvas = copy.deepcopy(table['img'])
    #     canvas[mask] = (0, 255, 0)
    #     image = image/2 + canvas/2
    #     cv2.imwrite(prefix+'_col_line_segm_%03d.png' % idx , cv2.resize(image, (image.shape[1]*4, image.shape[0]*4)))

    #     all_canvas[mask] = (0, 255, 0)
    # all_image = all_image/2 + all_canvas/2
    # cv2.imwrite(prefix+'_col_line_segm.png', cv2.resize(all_image, (all_image.shape[1]*4, all_image.shape[0]*4)))


def visualize_only(image, scale, pred_result, stride=4, back_color=(255,255,255), text_color=(255,0,0), line_width=10, 
    font_size=10, font_path='/train20/intern/zrzhang6/DocumentPretrain/dataprocess/process_rvlcdip/libs/simfang.ttf'):

    row_center_points, row_segm_logits, \
        col_center_points, col_segm_logits, \
            mg_logits, num_rows, num_cols = pred_result

    # draw grid polys
    scale = np.array(scale)[None, None, :]
    image_grid = copy.deepcopy(image)
    grid_polys, *_, row_segm_map, col_segm_map = parse_grid_bboxes([row_center_points], [row_segm_logits], \
        [col_center_points], [col_segm_logits], stride, stride)
    # print(grid_polys)
    for poly in grid_polys[0].cpu().numpy():
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        poly = np.array(poly).reshape(-1,1,2).astype('int')
        poly = (poly * scale).astype(np.int)
        # print(poly)
        cv2.polylines(image_grid, poly, True, color, line_width)

    # draw grid bboxes
    grid_bboxes = grid_polys[0].reshape(-1, 4, 2) # (N, 4, 2)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = [torch.stack([x1, y1, x2, y2], dim=-1)] # (N, 4)
    
    # parse table layout
    layout, spans = parse_layout(mg_logits, num_rows, num_cols)
    cells = parse_cells(layout, spans, grid_bboxes[0].cpu().numpy(), grid_polys[0].cpu().numpy())
    
    # draw cell poly
    image_cell = copy.deepcopy(image)
    for cell in cells:
        color = (random.randint(0, 255), random.randint(0, 255),
            random.randint(0, 255))
        poly = np.array(cell['segmentation']).reshape(-1,1,2).astype('int')
        poly =(poly * scale).astype(np.int)
        cv2.polylines(image_cell, [poly], True, (255,100,100), line_width)

    # image_grid = cv2.resize(image_grid, None, fx=0.25, fy=0.25)
    # image_cell = cv2.resize(image_cell, None, fx=0.25, fy=0.25)
    return image_grid, image_cell
   