from numpy import rec
import torch
import tqdm
import copy
import Polygon
import torch.nn.functional as F
import numpy as np
from .visualize import parse_grid_bboxes, trans2cellbbox, trans2cellpoly,trans2cellpoly_v1
from .scitsr.eval import json2Relations, eval_relations
import cv2
from collections import defaultdict
import shapely.geometry as SG
import shapely
def get_layout_spans(pred_result  ):
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    layout = get_layout(mg_logits)
    spans = layout2spans_v2(layout)
    return layout, spans

# 搜索一个合法解

def get_layout(mg_logits):
    id_result = torch.argsort( mg_logits, dim=0, descending=True )
    # scan from left to right, up to down
    id_max = torch.argmax(mg_logits,dim=0)
    id_max[0,0]=0
    mask = torch.zeros_like(id_max)
    num_row,num_col = id_max.shape
    index = 1
    for row in range(num_row):
        for col in range(num_col):
            if mask[row,col]!=0:
                continue
            if id_max[row,col]!=0:
                id_max[row,col]=0
            mask[row, col]=index
            
            end_col=col
            end_row=row

            for row_temp in range(row+1,num_row):
                if id_max[row_temp,col]==2: 
                    # mask[row_temp,col]=index
                    end_row=row_temp
                else:
                    break

            for col_temp in range(col+1, num_col):
                if id_max[row, col_temp]==1:
                    # mask[row,col_temp]=index
                    end_col=col_temp
                else:
                    break

            if torch.all( id_max[ row+1: end_row+1, col+1:end_col+1 ]==3 ):
                mask[ row: end_row+1, col:end_col+1 ]=index

            else:
                # L 的置信度更高
                end_row1 = end_row
                for row_temp in range(row+1,end_row+1):
                    if torch.all( id_max[ row+1: row_temp+1, col+1:end_col+1 ]==3 ):
                        end_row1=row_temp
                    else:
                        break
                
                end_col1 = end_col
                for col_temp in range(col+1,end_col+1):
                    if torch.all( id_max[ row+1: end_row+1, col+1:col_temp+1 ]==3 ):
                        end_col1=col_temp
                    else:
                        break
                
                # 看哪一个改变的更多
                if (end_col1-end_col)*(end_row-row)>(end_row1-end_row)*(end_col-col):
                    end_row = end_row1
                elif (end_col1-end_col)*(end_row-row)<(end_row1-end_row)*(end_col-col):
                    end_col=end_col1
                else:
                    end_row=end_row1

                mask[ row: end_row+1, col:end_col+1 ]=index
                
            index+=1
    # print(torch.min(mask)) 
    assert torch.min(mask)==1
    return (mask-1).cpu().numpy()
 


def get_layout_spans_tm(pred_result  ):
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits,mg_c_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    layout = get_layout_tm(mg_logits,mg_c_logits)
    spans = layout2spans_v2(layout)
    return layout, spans

# 搜索一个合法解

def get_layout_spans_tm_v1(pred_result  ):
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits,mg_c_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    layout = get_layout_tm_v1(mg_logits,mg_c_logits)
    spans = layout2spans_v2(layout)
    return layout, spans

def get_layout_tm(mg_logits,mg_c_logits):
    # 相信单独分支对于S的预测。
    # scan from left to right, up to down
    id_max = torch.argmax(mg_logits,dim=0)
    id_c_max = torch.argmin(mg_c_logits,dim=0) # 保证这个地方一定是C
    # id_ulx = torch.argmax( mg_logits[1:,...] )
    # id_max = (id_c_max+id_ulx)*id_c_max
    id_max = id_max*id_c_max
    id_max[0,0]=0
    mask = torch.zeros_like(id_max)
    num_row,num_col = id_max.shape
    index = 1
    for row in range(num_row):
        for col in range(num_col):
            if mask[row,col]!=0:
                continue
            if id_max[row,col]!=0:
                id_max[row,col]=0
            mask[row, col]=index
            
            end_col=col
            end_row=row

            for row_temp in range(row+1,num_row):
                if id_max[row_temp,col]==2: 
                    # mask[row_temp,col]=index
                    end_row=row_temp
                else:
                    break

            for col_temp in range(col+1, num_col):
                if id_max[row, col_temp]==1:
                    # mask[row,col_temp]=index
                    end_col=col_temp
                else:
                    break

            if torch.all( id_max[ row+1: end_row+1, col+1:end_col+1 ]==3 ):
                mask[ row: end_row+1, col:end_col+1 ]=index

            else:
                end_row1 = end_row
                for row_temp in range(row+1,end_row+1):
                    if torch.all( id_max[ row+1: row_temp+1, col+1:end_col+1 ]==3 ):
                        end_row1=row_temp
                    else:
                        break
                
                end_col1 = end_col
                for col_temp in range(col+1,end_col+1):
                    if torch.all( id_max[ row+1: end_row+1, col+1:col_temp+1 ]==3 ):
                        end_col1=col_temp
                    else:
                        break
                
                # 看哪一个改变的更多
                if (end_col1-end_col)*(end_row-row)>(end_row1-end_row)*(end_col-col):
                    end_row = end_row1
                elif (end_col1-end_col)*(end_row-row)<(end_row1-end_row)*(end_col-col):
                    end_col=end_col1
                else:
                    end_row=end_row1

                mask[ row: end_row+1, col:end_col+1 ]=index
                
            index+=1
    assert torch.min(mask)==1
    return (mask-1).cpu().numpy()

def get_layout_tm_v1(mg_logits,mg_c_logits):
    # 相信单独分支对于S的预测。
    # 其余地方，取不是S以外的最大action
    # scan from left to right, up to down
    # print(mg_logits.shape, mg_c_logits.shape)
    id_c_max = torch.argmin(mg_c_logits,dim=0) # 保证这个地方一定是C
    id_ulx = torch.argmax( mg_logits[1:,...],dim=0 )
    id_max = (id_c_max+id_ulx)*id_c_max
    id_max[0,0]=0
    mask = torch.zeros_like(id_max)
    num_row,num_col = id_max.shape
    index = 1
    for row in range(num_row):
        for col in range(num_col):
            if mask[row,col]!=0:
                continue
            if id_max[row,col]!=0:
                id_max[row,col]=0
            mask[row, col]=index
            
            end_col=col
            end_row=row

            for row_temp in range(row+1,num_row):
                if id_max[row_temp,col]==2: 
                    # mask[row_temp,col]=index
                    end_row=row_temp
                else:
                    break

            for col_temp in range(col+1, num_col):
                if id_max[row, col_temp]==1:
                    # mask[row,col_temp]=index
                    end_col=col_temp
                else:
                    break

            if torch.all( id_max[ row+1: end_row+1, col+1:end_col+1 ]==3 ):
                mask[ row: end_row+1, col:end_col+1 ]=index

            else:
                end_row1 = end_row
                for row_temp in range(row+1,end_row+1):
                    if torch.all( id_max[ row+1: row_temp+1, col+1:end_col+1 ]==3 ):
                        end_row1=row_temp
                    else:
                        break
                
                end_col1 = end_col
                for col_temp in range(col+1,end_col+1):
                    if torch.all( id_max[ row+1: end_row+1, col+1:col_temp+1 ]==3 ):
                        end_col1=col_temp
                    else:
                        break
                
                # 看哪一个改变的更多
                if (end_col1-end_col)*(end_row-row)>(end_row1-end_row)*(end_col-col):
                    end_row = end_row1
                elif (end_col1-end_col)*(end_row-row)<(end_row1-end_row)*(end_col-col):
                    end_col=end_col1
                else:
                    end_row=end_row1

                mask[ row: end_row+1, col:end_col+1 ]=index
                
            index+=1
    assert torch.min(mask)==1
    return (mask-1).cpu().numpy()


def get_layout_v1(mg_logits):
    # 修改生成不闭合单元格的现象
    id_result = torch.argsort( mg_logits, dim=0, descending=True )
    id_max = torch.argmax(mg_logits,dim=0)
    id_max[0,0]=0
    mask = torch.zeros_like(id_max)
    num_row,num_col = id_max.shape
    index = 1
    for row in range(num_row):
        for col in range(num_col):
            if mask[row,col]!=0:
                continue
            if id_max[row,col]!=0:
                id_max[row,col]=0
            mask[row, col]=index
            
            end_col=col
            end_row=row

            for row_temp in range(row+1,num_row):
                if id_max[row_temp,col]==2: 
                    # mask[row_temp,col]=index
                    end_row=row_temp
                else:
                    break

            for col_temp in range(col+1, num_col):
                if id_max[row, col_temp]==1:
                    # mask[row,col_temp]=index
                    end_col=col_temp
                else:
                    break

            if torch.all( id_max[ row+1: end_row+1, col+1:end_col+1 ]==3 ):
                mask[ row: end_row+1, col:end_col+1 ]=index

            else:
                # L 的置信度更高
                end_row1 = end_row
                for row_temp in range(row+1,end_row+1):
                    if torch.all( id_max[ row+1: row_temp+1, col+1:end_col+1 ]==3 ):
                        end_row1=row_temp
                    else:
                        break
                
                end_col1 = end_col
                for col_temp in range(col+1,end_col+1):
                    if torch.all( id_max[ row+1: end_row+1, col+1:col_temp+1 ]==3 ):
                        end_col1=col_temp
                    else:
                        break
                
                # 看哪一个改变的更多
                if (end_col1-end_col)*(end_row-row)>(end_row1-end_row)*(end_col-col):
                    end_row = end_row1
                elif (end_col1-end_col)*(end_row-row)<(end_row1-end_row)*(end_col-col):
                    end_col=end_col1
                else:
                    end_row=end_row1

                mask[ row: end_row+1, col:end_col+1 ]=index
                
            index+=1
    # print(torch.min(mask)) 
    assert torch.min(mask)==1
    return (mask-1).cpu().numpy()

def pred_result_to_table_new_token(table, pred_result,cfg=None):
    """先使用SEMv2_SciTSR得到layout、col_segment"""
    layout, spans = get_layout_spans(pred_result)
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,¥
        col_center_points, col_segm_logits, table_gird_bboxes)
    grids = list()
    for g in table_gird_bboxes.detach().cpu().numpy():
        grids.append( { "segmentation": [g] } )
    table = dict(
        layout=layout,
        cells=cells,
        grids=grids,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table   


def pred_result_to_table_new_token_iflytab(table, pred_result,cfg=None, is_infer=False):
    layout, spans = get_layout_spans(pred_result)
    row_center_points, row_segm_logits, col_center_points, col_segm_logits, mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,
        col_center_points, col_segm_logits, table_gird_bboxes)
    if not is_infer:
        extend_cell_lines(cells, table['line_polys'])
    grids = list()
    for g in table_gird_bboxes.detach().cpu().numpy():
        grids.append( { "segmentation": [g] } )
    table = dict(
        layout=layout,
        cells=cells,
        grids=grids,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table   


def pred_result_to_table_new_token_iflytab_tm(table, pred_result,cfg=None):

    layout, spans = get_layout_spans_tm(pred_result)
    row_center_points, row_segm_logits, col_center_points, col_segm_logits, mg_logits,mg_c_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,
        col_center_points, col_segm_logits, table_gird_bboxes)
    extend_cell_lines(cells, table['line_polys'])

    table = dict(
        layout=layout,
        cells=cells,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table   
 

def pred_result_to_table_new_token_iflytab_tm_v1(table, pred_result,cfg=None):
    
    layout, spans = get_layout_spans_tm_v1(pred_result)
    row_center_points, row_segm_logits, col_center_points, col_segm_logits, mg_logits,mg_c_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,col_center_points, col_segm_logits, table_gird_bboxes)
    extend_cell_lines(cells, table['line_polys'])

    table = dict(
        layout=layout,
        cells=cells,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table

 

def pred_result_to_table_new_token_raw(table, pred_result,cfg=None):
    layout, spans = get_layout_spans(pred_result)
    row_center_points, row_segm_logits, col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits, table_gird_bboxes)
    grids = list()
    for g in table_gird_bboxes.detach().cpu().numpy():
        grids.append( { "segmentation": [g] } )
    table = dict(
        layout=layout,
        cells=cells,
        grids=grids,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table 

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
        # try:
        #     assert np.all(layout[y1:y2, x1:x2] == cell_id)
        # except:
        #     import pdb; pdb.set_trace()
        cells_span.append([x1, y1, x2, y2])
    return cells_span


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


def parse_cells_gt(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits):
    grid_polys, *_ = parse_grid_bboxes([row_center_points], [row_segm_logits], \
        [col_center_points], [col_segm_logits], stride_w=4, stride_h=4, score_threshold=0.25)
    grid_polys = grid_polys[0]
    grid_bboxes = grid_polys.reshape(-1, 4, 2) # (N, 4, 2)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.cpu().numpy().astype('int64')
    grid_bboxes = grid_bboxes.cpu().numpy().astype('int64')
    
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
            row_end_idx=int(span[3]),
            transcript=''
        )
        cells.append(cell)

    return cells

def parse_cells(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits, grid_polys):
    
    grid_polys_change = grid_polys.clone()
    grid_polys_change[:,2:4] = grid_polys[:,6:8]
    grid_polys_change[:,6:8] = grid_polys[:,2:4]

    grid_bboxes = grid_polys.reshape(-1, 4, 2) # (N, 4, 2) 不规则四边形
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4) # 外接矩形

    grid_polys = grid_polys.cpu().numpy().astype('int64') #N*8
    grid_polys_change = grid_polys_change.cpu().numpy().astype('int64') #N*8
    grid_bboxes = grid_bboxes.cpu().numpy().astype('int64')
    
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys_change[cell_positions[:, 0]]
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        # print(row_indexs)
        # print(col_indexs)
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_rect = bbox2rect(cell_bbox)

        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            rect = cell_rect,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3]),
            transcript=''
        )
        cells.append(cell)

    return cells
 
def cal_cell_rectangles(cells, is_need_rect, is_need_rotaterect):
    if not (is_need_rect or is_need_rotaterect):
        return
    for item in cells:
        if is_need_rect:
            item["segmentation_rect"] = [[[item["bbox"][0],item["bbox"][1]], [item["bbox"][2],item["bbox"][1]], [item["bbox"][2],item["bbox"][3]],[item["bbox"][0],item["bbox"][3]]]]
        if is_need_rotaterect:
            segment = np.expand_dims(np.array(item["segmentation"], dtype=np.int32)[0], axis=1)
            (x,y),(w,h),a = cv2.minAreaRect(segment)
            rect_r = ((x,y), (w,h), a)
            item["segmentation_rotaterect"] = [cv2.boxPoints(rect_r).astype(np.int32).tolist()]
            # print(item["segmentation_rotaterect"])

    

def extend_cell_lines(cells, lines):
    # the transcript of correctly predicted cell is the gt text line box id
    # maybe one predicted cell cover/spaned more than one text line box
    # the step for TED, which takes the cell text content into account.
    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    lines = copy.deepcopy(lines)
    lines = [line.reshape(-1,2) for line in lines]

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    # lines_poly = [segmentation_to_polygon(item['segmentation']) for item in lines]
    lines_poly = [segmentation_to_polygon([item]) for item in lines]

    for line_idx, line_poly in enumerate(lines_poly):
        if line_poly.area() == 0:
            continue
        line_area = line_poly.area()
        max_overlap = 0
        max_overlap_idx = None
        for cell_idx, cell_poly in enumerate(cells_poly):
            overlap = (cell_poly & line_poly).area() / line_area
            if overlap > max_overlap:
                max_overlap_idx = cell_idx
                max_overlap = overlap
            if overlap > 0.99:
                break
        if max_overlap > 0:
            cells[max_overlap_idx]['transcript'] += str(line_idx) + '-'


def segmentation_to_bbox(segmentation):
    x1 = min([min([pt[0] for pt in contour]) for contour in segmentation])
    y1 = min([min([pt[1] for pt in contour]) for contour in segmentation])
    x2 = max([max([pt[0] for pt in contour]) for contour in segmentation])
    y2 = max([max([pt[1] for pt in contour]) for contour in segmentation])
    return [x1, y1, x2, y2]


def cal_cell_spans(table):
    layout = table['layout']
    num_cells = len(table['cells'])
    cells_span = list()
    for cell_id in range(num_cells):
        cell_positions = np.argwhere(layout == cell_id)
        y1 = np.min(cell_positions[:, 0])
        y2 = np.max(cell_positions[:, 0])
        x1 = np.min(cell_positions[:, 1])
        x2 = np.max(cell_positions[:, 1])
        # assert np.all(layout[y1:y2, x1:x2] == cell_id)
        cells_span.append([x1, y1, x2, y2])
    return cells_span


def trans_pred_result_semv2_2_semv1(pred_result, return_spans=False):
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result

    layout, spans = parse_layout_v2(mg_logits, num_rows, num_cols, score_threshold=0.5)
    
    return layout, spans


def process_layout_v2(score, index, use_score=False, is_merge=True, score_threshold=0.5): # 寻找连通块
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
            if torch.all(index[h_min:h_max+1, w_min:w_max+1] == id): # Connected
                layout_mask[h_min:h_max+1, w_min:w_max+1] = 1
                layout[h_min:h_max+1, w_min:w_max+1] = cell_id
            else:
                lf_row = crow
                lf_col = ccol
                col_mem = -1
                for col_ in range(lf_col, w_max + 1):
                    if index[lf_row, col_] == id: # how wide the cell spans
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



def layout2spans_v2(layout):
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



def parse_layout_v2(mg_logits, num_rows, num_cols, score_threshold=0.5):
    num_grids = int(num_rows) * int(num_cols)
    mg_probs = mg_logits[:num_grids, :int(num_rows), :int(num_cols)].sigmoid() # (N, H, W)
    _, indices = (mg_probs > score_threshold).float().cpu().max(dim=0) # (H, W)
    values, _ = mg_probs.max(dim=0) # (H, W)
    
    layout = process_layout_v2(values, indices, use_score=True, is_merge=False, score_threshold=score_threshold) # 根据分数确定最大连通区域
    layout = process_layout_v2(values, layout)
    layout = layout.cpu().numpy()
    spans = layout2spans_v2(layout)
    return layout, spans


def extend_content(cells, gt_cells):

    def segmentation_to_polygon(segmentation):
        polygon = Polygon.Polygon()
        for contour in segmentation:
            polygon = polygon + Polygon.Polygon(contour)
        return polygon

    cells_poly = [segmentation_to_polygon(item['segmentation']) for item in cells]
    gt_cells_poly = [segmentation_to_polygon(item['segmentation']) for item in gt_cells]

    for cell_idx, cell_poly in enumerate(cells_poly):
        max_overlap = 0
        max_overlap_idx = None
        for gt_idx, gt_poly in enumerate(gt_cells_poly):
            overlap = (cell_poly & gt_poly).area() / (cell_poly.area() + gt_poly.area() + 1e-5)
            if overlap > max_overlap:
                max_overlap_idx = gt_idx
                max_overlap = overlap
        if max_overlap > 0:
            cells[cell_idx]['transcript'] = gt_cells[max_overlap_idx]['transcript']
        else:
            cells[cell_idx]['transcript'] = "None"
    return cells

def match_segment_spans(segments, begins, ends):
    matched_segments = list()
    matched_spans = list()

    for segment_idx, segment in enumerate(segments):
        for span_idx, span in enumerate(zip(begins, ends)):

            if (span_idx not in matched_spans):
                if (segment >= span[0][0]) and (segment < span[1][0]):
                    matched_segments.append(segment_idx)
                    matched_spans.append(span_idx)
                    break

    return matched_segments, matched_spans

def cal_segment_pr(point_c, begins, ends):
    correct_nums = 0
    segment_nums = 0
    span_nums = 0
    matched_segments_idx, matched_spans = match_segment_spans(point_c, begins, ends)
        # unmatched_segments_idx = find_unmatch_segment_spans(pred_segments_pi, fg_spans_pi + bg_spans_pi)
    # print(point_c)
    correct_nums += len(matched_segments_idx)
    segment_nums += len(point_c) #  - len(unmatched_segments_idx) 
    span_nums += len(begins)
    # print(correct_nums, segment_nums, span_nums)
    return correct_nums, segment_nums, span_nums,matched_segments_idx, matched_spans

def cal_line_distance( pred_grids, target_grids, pred_num_rows, target_num_rows,  pred_num_cols, target_num_cols, pred_row_center_points, pred_col_center_points, target_row_cneter_boxes, target_col_center_boxes):
    ''' num_rows num_cols 是实际上的行列，不是分割线的数量 '''
    #找出匹配上的cneter point
    _, y1, _, y2 = np.split(target_row_cneter_boxes, 4, axis=-1)
    correct_nums, segment_nums, span_nums, row_matched_segments_idx, row_matched_spans = cal_segment_pr( pred_row_center_points, y1,y2 )
    row_det_p = correct_nums/(1e-5+segment_nums)
    row_det_r = correct_nums/(1e-5+span_nums)

    x1, _, x2, _ = np.split(target_col_center_boxes, 4, axis=-1)
    correct_nums, segment_nums, span_nums, col_matched_segments_idx, col_matched_spans = cal_segment_pr( pred_col_center_points, x1, x2 )
    col_det_p = correct_nums/(1e-5+segment_nums)
    col_det_r = correct_nums/(1e-5+span_nums)

    # 解析出每一条线# 挑选出匹配上的线
    pred_grids_nm = pred_grids.reshape((pred_num_rows, pred_num_cols, 8))
    # 预测出来的grid points的顺序是 lt,lb,rb,rt
    pred_line = np.zeros( (pred_num_rows+1, pred_num_cols+1, 2) )
    pred_line[:pred_num_rows ,:pred_num_cols ] = pred_grids_nm[ :pred_num_rows , :pred_num_cols, 0:2 ] # lt
    pred_line[:pred_num_rows , pred_num_cols:] = pred_grids_nm[  :pred_num_rows, pred_num_cols-1:, 6:8 ] # pred_grids_nm[ :pred_num_rows ,  pred_num_cols-1:, 2:4 ] 
    pred_line[ pred_num_rows:,:pred_num_cols ] = pred_grids_nm[  pred_num_rows-1:,   :pred_num_cols, 2:4 ]#pred_grids_nm[  pred_num_rows-1:, :pred_num_cols, 6:8 ]
    pred_line[ pred_num_rows , pred_num_cols ] = pred_grids_nm[  pred_num_rows-1,   pred_num_cols-1, 4:6 ]
    pred_line_matched = pred_line[row_matched_segments_idx][:, col_matched_segments_idx]
    # print(row_matched_segments_idx,col_matched_segments_idx)
    target_grids_nm = target_grids.reshape(( target_num_rows, target_num_cols, 8 ))
    target_line = np.zeros( (target_num_rows+1, target_num_cols+1, 2) )
    target_line[:target_num_rows, :target_num_cols ] = target_grids_nm[ :target_num_rows, :target_num_cols, 0:2 ]
    target_line[:target_num_rows,  target_num_cols:] = target_grids_nm[ :target_num_rows,  target_num_cols-1:, 2:4 ]
    target_line[ target_num_rows:,:target_num_cols ] = target_grids_nm[  target_num_rows-1:,:target_num_cols, 6:8 ]
    target_line[ target_num_rows,  target_num_cols ] = target_grids_nm[  target_num_rows-1,target_num_cols-1, 4:6 ]
    target_line_matched = target_line[ row_matched_spans][:, col_matched_spans ]
    # print(row_matched_spans,col_matched_spans)
    # print(row_det_p, row_det_r, col_det_p, col_det_r )
    # 隶｡邂苓ｷ晉ｦｻ
    line_distance = 0
    line_distance = np.sqrt(np.sum(np.square(pred_line_matched-target_line_matched),axis=2)).mean()
    return line_distance, row_det_p, row_det_r, col_det_p, col_det_r 
    
def pred_result_to_table(table, pred_result,cfg=None):
    layout, spans = trans_pred_result_semv2_2_semv1(pred_result, return_spans=True)
    row_center_points, row_segm_logits, \
            col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = pred_result
    
    cells = parse_cells(layout, spans, row_center_points, row_segm_logits,\
        col_center_points, col_segm_logits, table_gird_bboxes)
    extend_cell_lines(cells, table['line_polys'])
    if not cfg is None:
        cal_cell_rectangles( cells, cfg.is_need_rect, cfg.is_need_rotaterect )
    grids = list()
    for g in table_gird_bboxes.detach().cpu().numpy():
        grids.append( { "segmentation": [g] } )
    table = dict(
        layout=layout,
        cells=cells,
        grids=grids,
        grid_polys = table_gird_bboxes.detach().cpu().numpy(),
        num_row = layout.shape[0],
        num_col = layout.shape[1]
    )
    return table   
 
def gt_format_v1(table, cfg=None):
    # 逶ｴ謗･菴ｿ逕ｨline poly菴應ｸｺ
    row_start_bboxes = table['row_start_bboxes']
    col_start_bboxes = table['col_start_bboxes']
    row_line_masks = table['row_line_masks']
    col_line_masks = table['col_line_masks']
    grid_polys = list()
    grids = list()
    yp = (row_start_bboxes[:, 1] + row_start_bboxes[:, 3]) / 2
    xp = (col_start_bboxes[:, 0] + col_start_bboxes[:, 2]) / 2
    num_rows = len(row_start_bboxes)
    num_cols = len(col_start_bboxes)
    grid_points = np.zeros((num_rows, num_cols,2))
    grid_polys = list()
        
    for row_idx in range(num_rows):
        for col_idx in range(num_cols ):
            x1_segm_map = col_line_masks[col_idx] # (H, W)
            y1_segm_map = row_line_masks[row_idx] # (H, W)
            lt_segm_map = x1_segm_map + y1_segm_map # (H, W)
            y_lt, x_lt = np.where(lt_segm_map==2)
            if len(y_lt) > 0 and len(x_lt) > 0:
                x_lt = int(x_lt.mean())
                y_lt = int(y_lt.mean())
            else:
                x_lt = int(xp[col_idx])
                y_lt = int(yp[row_idx])

            grid_points[row_idx, col_idx] = np.array([x_lt, y_lt])
    grid_points_polys = np.zeros((num_rows-1,num_cols-1,8))
    grid_points_polys[:,:,0:2] = grid_points[:-1,:-1,:]
    grid_points_polys[:,:,2:4] = grid_points[1:,:-1,:]
    grid_points_polys[:,:,4:6] = grid_points[1:,1:,:]
    grid_points_polys[:,:,6:8] = grid_points[:-1,1:,:]
    grid_polys = grid_points_polys.reshape((-1,8)).tolist() # 注意这里顶点的顺序是从左上角开始逆时针排列的
    
    layout = table['layout']
    spans = layout2spans(layout)
    grid_polys = np.array(grid_polys).reshape(-1, 4, 2) # (N, 4, 2)
    grid_bboxes = torch.tensor(grid_polys)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.astype('int64')
    grid_bboxes = grid_bboxes.numpy().astype('int64')
    
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(int(num_cells)):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys[cell_positions[:, 0]]
        
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_rect = bbox2rect(cell_bbox)
        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            rect=cell_rect,
            segmentation=cell_poly,
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3]),
        )
        cells.append(cell)
    table = dict(
        layout=layout,
        cells=cells,
        image_path = table['image_path'],
        num_row = len(row_start_bboxes)-1,
        num_col = len(col_start_bboxes)-1
    )
    return table

def bbox2rect(bbox):
    x1,y1,x2,y2 = bbox
    rect = [x1,y1,x2,y1,x2,y2,x1,y2]
    return rect
 
def gt_format(table, cfg=None):
    row_start_bboxes = table['row_start_bboxes']
    col_start_bboxes = table['col_start_bboxes']
    row_line_masks = table['row_line_masks']
    col_line_masks = table['col_line_masks']
    grid_polys = list()
    grids = list()
    yp = (row_start_bboxes[:, 1] + row_start_bboxes[:, 3]) / 2
    xp = (col_start_bboxes[:, 0] + col_start_bboxes[:, 2]) / 2
    for row_idx in range(len(row_start_bboxes)-1):
        for col_idx in range(len(col_start_bboxes)-1):
            x1_segm_map = col_line_masks[col_idx] # (H, W)
            y1_segm_map = row_line_masks[row_idx] # (H, W)
            x2_segm_map = col_line_masks[col_idx+1] # (H, W)
            y2_segm_map = row_line_masks[row_idx+1] # (H, W)
            
            lt_segm_map = x1_segm_map * y1_segm_map # (H, W)
            y_lt, x_lt = np.where(lt_segm_map==1)
            if len(y_lt) > 0:
                y_lt = int(y_lt.mean())
            else:
                y_lt = int(yp[row_idx])
            if len(x_lt) > 0:
                x_lt = int(x_lt.mean())
            else:
                x_lt = int(xp[col_idx])

            rt_segm_map = x2_segm_map * y1_segm_map # (H, W)
            y_rt, x_rt = np.where(rt_segm_map==1)
            if len(y_rt) > 0:
                y_rt = int(y_rt.mean())
            else:
                y_rt = int(yp[row_idx])
            if len(x_rt) > 0:
                x_rt = int(x_rt.mean())
            else:
                x_rt = int(xp[col_idx+1])
                
            rb_segm_map = x2_segm_map * y2_segm_map # (H, W)
            y_rb, x_rb = np.where(rb_segm_map==1)
            if len(y_rb) > 0:
                y_rb = int(y_rb.mean())
            else:
                y_rb = int(yp[row_idx+1])
            if len(x_rb) > 0:
                x_rb = int(x_rb.mean())
            else:
                x_rb = int(xp[col_idx+1])
                
            lb_segm_map = x1_segm_map * y2_segm_map # (H, W)
            y_lb, x_lb = np.where(lb_segm_map==1)
            if len(y_lb) > 0:
                y_lb = int(y_lb.mean())
            else:
                y_lb = int(yp[row_idx+1])
            if len(x_lb) > 0:
                x_lb = int(x_lb.mean())
            else:
                x_lb = int(xp[col_idx])
            grids.append({"segmentation":[ [x_lt, y_lt, x_rt, y_rt, x_rb, y_rb, x_lb, y_lb] ]})
            grid_polys.append([x_lt, y_lt, x_rt, y_rt, x_rb, y_rb, x_lb, y_lb])
    
    layout = table['layout']
    spans = layout2spans(layout)
    grid_polys = np.array(grid_polys).reshape(-1, 4, 2) # (N, 4, 2)
    grid_bboxes = torch.tensor(grid_polys)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.astype('int64')
    grid_bboxes = grid_bboxes.numpy().astype('int64')
    
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(int(num_cells)):
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
            row_end_idx=int(span[3]),
            transcript=''
        )
        cells.append(cell)

    extend_cell_lines(cells, table['line_polys'])
    if not cfg is None:
        cal_cell_rectangles( cells, cfg.is_need_rect, cfg.is_need_rotaterect )
    table = dict(
        img=table['img'],
        layout=layout,
        cells=cells,
        grids=grids,
        image_path = table['image_path'],
        line_polys=table['line_polys'],
        row_line_masks=table['row_line_masks'],
        col_line_masks=table['col_line_masks'],
        grid_polys = grid_polys,
        num_row = len(row_start_bboxes)-1,
        num_col = len(col_start_bboxes)-1
    )
    return table


def parse_grid_bboxes_delta_np(row_center_points, row_deltas, col_center_points, col_deltas,\
    stride_w, stride_h, padding_shape, insert_rows=[], insert_cols=[], score_threshold=0.5, kernel_size=3, radius=30):
    '''
        parse the start bboxes and segmentation results to table grid bboxes
    '''

    batch_grid_bboxes = []
    batch_num_rows = []
    batch_num_cols = []
    pad_h, pad_w = padding_shape[-2:]
    for batch_idx in range(len(row_center_points)):
        # parse start point
        rs_yc = row_center_points[batch_idx] # (NumRow, 1)
        cs_xc = col_center_points[batch_idx]# (NumCol, 1)
        insert_yc = insert_rows[batch_idx][0]
        insert_xc = insert_cols[batch_idx][0]
        rs_yc = torch.sort(torch.cat((rs_yc, insert_yc), dim=0))[0]
        cs_xc = torch.sort(torch.cat((cs_xc, insert_xc), dim=0))[0]
        row_deltas_batch = row_deltas[batch_idx]
        col_deltas_batch = col_deltas[batch_idx]
        row_delta_len = row_deltas[batch_idx].shape[-1]
        for row_ in rs_yc: # 额外的地方补成零，需要将这荳ｪ螳樔ｾ区薯蜈･
            if row_ in insert_yc:
                insert_index = torch.where(rs_yc==row_)[0][-1]
                y = torch.zeros((1,row_delta_len)).to(row_deltas[batch_idx]).reshape((1,-1))
                row_deltas_batch = torch.cat( [ row_deltas_batch[:insert_index, : ], y, row_deltas_batch[insert_index:,:] ], dim=0)
        col_delta_len = col_deltas[batch_idx].shape[-1]
        for col_ in cs_xc:
            if col_ in insert_xc:
                insert_index = torch.where(cs_xc==col_)[0][-1]
                x = torch.zeros((1,col_delta_len)).to(col_deltas[batch_idx]).reshape((1,-1))
                col_deltas_batch = torch.cat( [ col_deltas_batch[:insert_index, :], x, col_deltas_batch[insert_index:] ], dim=0)
        # parse the poly bbox
        num_rows = rs_yc.size(0)
        num_cols = cs_xc.size(0)
        grid_points = np.zeros((num_rows, num_cols,2))
        grid_polys = list()
        for row_idx in range(num_rows):
            for col_idx in range(num_cols):
                row_deltas_temp = row_deltas_batch[row_idx].detach().cpu().numpy() # 是否要进行平滑？
                row_y = row_deltas_temp*15 + np.ones_like(row_deltas_temp)*rs_yc[row_idx].item()

                row_x = np.arange(0,len(row_y))*32
                row_points = np.stack([row_x,row_y],axis=-1).astype(np.int32)
                row_points = np.append( row_points, np.array([[pad_w-1, row_y[-1]]]).astype(np.int32),axis=0 )
                row_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
                row_canve = cv2.polylines( row_canve, [ row_points ],False, 1, 5 )

                col_deltas_temp = col_deltas_batch[col_idx].detach().cpu().numpy() # 是否要进行平滑？
                col_x = col_deltas_temp*15 + np.ones_like(col_deltas_temp)*cs_xc[col_idx].item()
                col_y = np.arange(0,len(col_x))*32
                col_points = np.stack([col_x,col_y],axis=-1).astype(np.int32)
                col_points = np.append( col_points, np.array([[col_x[-1],pad_h-1]]).astype(np.int32),axis=0 )
                col_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
                col_canve = cv2.polylines( col_canve, [ col_points ],False, 1, 5 )
                
                y_lt1, x_lt1 = np.where(row_canve+col_canve==2)
                if len(y_lt1) > 0 and len(x_lt1) > 0:
                    x_lt1 = int(x_lt1.mean() )
                    y_lt1 = int(y_lt1.mean() )
                else:
                    x_lt1 = int(cs_xc[col_idx])
                    y_lt1 = int(rs_yc[row_idx])
                    
                grid_points[row_idx, col_idx] = np.array([x_lt1, y_lt1])
                
        grid_points_polys = np.zeros((num_rows-1,num_cols-1,8))
        grid_points_polys[:,:,0:2] = grid_points[:-1,:-1,:]
        grid_points_polys[:,:,2:4] = grid_points[1:,:-1,:]
        grid_points_polys[:,:,4:6] = grid_points[1:,1:,:]
        grid_points_polys[:,:,6:8] = grid_points[:-1,1:,:]
        grid_polys = grid_points_polys.reshape((-1,8)).tolist()


        if len(grid_polys) == 0:
            grid_polys.append([0, 0, 0, 0, 0, 0, 0, 0])
            num_cols = 2
            num_rows = 2

        grid_polys = torch.tensor(grid_polys, dtype=torch.float, device=row_deltas[0].device)
        batch_grid_bboxes.append(grid_polys)
        batch_num_cols.append(num_cols-1)
        batch_num_rows.append(num_rows-1)
    return batch_grid_bboxes, batch_num_rows, batch_num_cols

def gt_format_delta(table, cfg=None):
    row_start_bboxes = table['row_start_bboxes']
    col_start_bboxes = table['col_start_bboxes']
    pad_h = table["image_h"]
    pad_w = table["image_w"]

    grid_polys = list()
    grids = list()
    yp = (row_start_bboxes[:, 1] + row_start_bboxes[:, 3]) / 2
    xp = (col_start_bboxes[:, 0] + col_start_bboxes[:, 2]) / 2
     # parse start point
    rs_yc = yp
    cs_xc = xp
    col_deltas_batch = table["col_deltas"]
    row_deltas_batch = table["row_deltas"]
    # print(col_deltas_batch)
    # parse the poly bbox
    num_rows = rs_yc.shape[0]
    num_cols = cs_xc.shape[0]

    grid_points = np.zeros((num_rows, num_cols,2))
    grid_polys = list()
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            row_deltas_temp = row_deltas_batch[row_idx] # 是否要进行平滑？
            row_y = row_deltas_temp + np.ones_like(row_deltas_temp)*rs_yc[row_idx]

            row_x = np.arange(0,len(row_y))
            row_points = np.stack([row_x,row_y],axis=-1).astype(np.int32)
            row_points = np.append( row_points, np.array([[pad_w-1, row_y[-1]]]).astype(np.int32),axis=0 )
            row_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
            row_canve = cv2.polylines( row_canve, [ row_points ],False, 1, 5 )

            col_deltas_temp = col_deltas_batch[col_idx] # 是否要进行平滑？
            col_x = col_deltas_temp + np.ones_like(col_deltas_temp)*cs_xc[col_idx]
            col_y = np.arange(0,len(col_x))
            col_points = np.stack([col_x,col_y],axis=-1).astype(np.int32)
            col_points = np.append( col_points, np.array([[col_x[-1],pad_h-1]]).astype(np.int32),axis=0 )
            col_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
            col_canve = cv2.polylines( col_canve, [ col_points ],False, 1, 5 )
            
            y_lt1, x_lt1 = np.where(row_canve+col_canve==2)
            if len(y_lt1) > 0 and len(x_lt1) > 0:
                x_lt1 = int(x_lt1.mean() )
                y_lt1 = int(y_lt1.mean() )
            else:
                x_lt1 = int(cs_xc[col_idx])
                y_lt1 = int(rs_yc[row_idx])
                
            grid_points[row_idx, col_idx] = np.array([x_lt1, y_lt1])
            
    grid_points_polys = np.zeros((num_rows-1,num_cols-1,8))
    grid_points_polys[:,:,0:2] = grid_points[:-1,:-1,:]
    grid_points_polys[:,:,2:4] = grid_points[1:,:-1,:]
    grid_points_polys[:,:,4:6] = grid_points[1:,1:,:]
    grid_points_polys[:,:,6:8] = grid_points[:-1,1:,:]
    grid_polys = grid_points_polys.reshape((-1,8)).tolist()
    for g in grid_polys:
        grids.append( { "segmentation": [g] } )
    
    grid_polys_change = np.array(grid_polys).copy()
    grid_polys_change[:,2:4] = np.array(grid_polys)[:,6:8]
    grid_polys_change[:,6:8] = np.array(grid_polys)[:,2:4]

    layout = table['layout']
    spans = layout2spans(layout)
    grid_polys = np.array(grid_polys).reshape(-1, 4, 2) # (N, 4, 2)
    grid_bboxes = torch.tensor(grid_polys)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.astype('int64')
    grid_bboxes = grid_bboxes.numpy().astype('int64')
    
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(int(num_cells)):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys_change[cell_positions[:, 0]]
        
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        # print(row_indexs)
        # print(col_indexs)
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_rect = bbox2rect(cell_bbox)
        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            rect=cell_rect,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3]),
            transcript=''
        )
        cells.append(cell)

    extend_cell_lines(cells, table['line_polys'])
    if not cfg is None:
        cal_cell_rectangles( cells, cfg.is_need_rect, cfg.is_need_rotaterect )
    table = dict(
        img=table['img'],
        layout=layout,
        cells=cells,
        grids=grids,
        image_path = table['image_path'],
        line_polys=table['line_polys'],
        grid_polys = grid_polys,
        num_row = len(row_start_bboxes)-1,
        num_col = len(col_start_bboxes)-1
    )
    return table
  


def gt_format_delta_iflytab(table, cfg=None):
    row_start_bboxes = table['row_start_bboxes']
    col_start_bboxes = table['col_start_bboxes']
    pad_h = table["image_h"]
    pad_w = table["image_w"]

    grid_polys = list()
    grids = list()
    yp = (row_start_bboxes[:, 1] + row_start_bboxes[:, 3]) / 2
    xp = (col_start_bboxes[:, 0] + col_start_bboxes[:, 2]) / 2
     # parse start point
    rs_yc = yp
    cs_xc = xp
    col_deltas_batch = table["col_deltas"]
    row_deltas_batch = table["row_deltas"]
    # parse the poly bbox
    num_rows = rs_yc.shape[0]
    num_cols = cs_xc.shape[0]

    grid_points = np.zeros((num_rows, num_cols,2))
    grid_polys = list()
    for row_idx in range(num_rows):
        for col_idx in range(num_cols):
            row_deltas_temp = row_deltas_batch[row_idx] # 是否要进行平滑？
            row_y = row_deltas_temp + np.ones_like(row_deltas_temp)*rs_yc[row_idx]

            row_x = np.arange(0,len(row_y))
            row_points = np.stack([row_x,row_y],axis=-1).astype(np.int32)
            row_points = np.append( row_points, np.array([[pad_w-1, row_y[-1]]]).astype(np.int32),axis=0 )
            row_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
            row_canve = cv2.polylines( row_canve, [ row_points ],False, 1, 5 )

            col_deltas_temp = col_deltas_batch[col_idx] # 是否要进行平滑？
            col_x = col_deltas_temp + np.ones_like(col_deltas_temp)*cs_xc[col_idx]
            col_y = np.arange(0,len(col_x))
            col_points = np.stack([col_x,col_y],axis=-1).astype(np.int32)
            col_points = np.append( col_points, np.array([[col_x[-1],pad_h-1]]).astype(np.int32),axis=0 )
            col_canve = np.zeros((pad_h, pad_w), dtype=np.uint8)
            col_canve = cv2.polylines( col_canve, [ col_points ],False, 1, 5 )
            
            y_lt1, x_lt1 = np.where(row_canve+col_canve==2)
            if len(y_lt1) > 0 and len(x_lt1) > 0:
                x_lt1 = int(x_lt1.mean() )
                y_lt1 = int(y_lt1.mean() )
            else:
                x_lt1 = int(cs_xc[col_idx])
                y_lt1 = int(rs_yc[row_idx])
                
            grid_points[row_idx, col_idx] = np.array([x_lt1, y_lt1])
            
    grid_points_polys = np.zeros((num_rows-1,num_cols-1,8))
    grid_points_polys[:,:,0:2] = grid_points[:-1,:-1,:]
    grid_points_polys[:,:,2:4] = grid_points[1:,:-1,:]
    grid_points_polys[:,:,4:6] = grid_points[1:,1:,:]
    grid_points_polys[:,:,6:8] = grid_points[:-1,1:,:]
    grid_polys = grid_points_polys.reshape((-1,8)).tolist()
    # grids = list()
    for g in grid_polys:
        grids.append( { "segmentation": [g] } )
    grid_polys_change = np.array(grid_polys).copy()
    grid_polys_change[:,2:4] = np.array(grid_polys)[:,6:8]
    grid_polys_change[:,6:8] = np.array(grid_polys)[:,2:4]

    layout = table['layout']
    spans = layout2spans(layout)
    grid_polys = np.array(grid_polys).reshape(-1, 4, 2) # (N, 4, 2)
    grid_bboxes = torch.tensor(grid_polys)
    x1 = grid_bboxes[:, :, 0].min(-1)[0] # (N)
    y1 = grid_bboxes[:, :, 1].min(-1)[0] # (N)
    x2 = grid_bboxes[:, :, 0].max(-1)[0] # (N)
    y2 = grid_bboxes[:, :, 1].max(-1)[0] # (N)
    grid_bboxes = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)

    grid_polys = grid_polys.astype('int64')
    grid_bboxes = grid_bboxes.numpy().astype('int64')
    
    cells = list()
    num_cells = np.max(layout) + 1
    for cell_id in range(int(num_cells)):
        cell_positions = np.argwhere(layout.reshape(-1)==cell_id)
        valid_grid_bboxes = grid_bboxes[cell_positions[:, 0]]
        valid_grid_polys = grid_polys_change[cell_positions[:, 0]]
        
        row_indexs = cell_positions[:,0]// layout.shape[1]
        col_indexs = cell_positions[:,0]-row_indexs*layout.shape[1]
        # print(row_indexs)
        # print(col_indexs)
        cell_bbox = trans2cellbbox(valid_grid_bboxes)
        cell_rect = bbox2rect(cell_bbox)
        cell_poly = trans2cellpoly_v1(valid_grid_polys, row_indexs, col_indexs)

        span = spans[cell_id]

        cell = dict(
            bbox=cell_bbox,
            rect=cell_rect,
            segmentation=[cell_poly],
            col_start_idx=int(span[0]),
            row_start_idx=int(span[1]),
            col_end_idx=int(span[2]),
            row_end_idx=int(span[3]),
            transcript=''
        )
        cells.append(cell)

    extend_cell_lines(cells, table['line_polys'])
    if not cfg is None:
        cal_cell_rectangles( cells, cfg.is_need_rect, cfg.is_need_rotaterect )
    table = dict(
        img=table['img'],
        layout=layout,
        cells=cells,
        grids=grids,
        image_path = table['image_path'],
        line_polys=table['line_polys'],
        grid_polys = grid_polys,
        num_row = len(row_start_bboxes)-1,
        num_col = len(col_start_bboxes)-1
    )
    return table
  


def table_to_relations(table):
    cell_spans = cal_cell_spans(table)
    contents = [''.join(cell['transcript']).split() for cell in table['cells']]
    relations = []
    for span, content in zip(cell_spans, contents):
        x1, y1, x2, y2 = span
        relations.append(dict(start_row=y1, end_row=y2, start_col=x1, end_col=x2, content=content))
    return dict(cells=relations)

def cal_cell_iou(label, pred, type_list):
    ious = dict()
    for type_i in type_list:
        poly_gt = Polygon.Polygon( label[type_i][0] )
        poly_pred = Polygon.Polygon( pred[type_i][0] )
        inter = poly_pred & poly_gt
        if len(inter) == 0:
            ious[type_i] = 0
        else:
            inter_area = inter.area()
            union = poly_pred.area() + poly_gt.area() - inter_area
            
            try:
                iou = inter_area / union
            except:
                iou = 0
            ious[type_i] = iou
    return ious

def bbox_iou_eval(label, pred, type_list):
    ious = dict()
    for type_i in type_list:
        box1 = np.array(label[type_i][0]).reshape(-1, 2)
        poly1 = SG.Polygon(box1).convex_hull #POLYGON ((0 0, 0 2, 2 2, 2 0, 0 0))

        box2 = np.array(pred[type_i][0]).reshape(-1, 2)
        poly2 = SG.Polygon(box2).convex_hull

        if not poly1.intersects(poly2):  # 螯よ棡荳､蝗幄ｾｹ蠖｢荳咲嶌莠､
            iou = 0
        else:
            try:
                inter_area = poly1.intersection(poly2).area  # 逶ｸ莠､髱｢遘ｯ
                iou = float(inter_area) / (poly1.area + poly2.area - inter_area)
            except:
                # print('iou set to 0')
                iou = 0
        ious[type_i] = iou
    return ious

def eval_cell_iou(label, pred, iou_list, type_list):
    # print("label",label)
    len_label = len(label)
    len_pred = len(pred)
    iou2id = { iou:id for id,iou in enumerate((iou_list)) }
    type2id = { type_i:id for id,type_i in enumerate((type_list)) }
    label_flag = np.zeros((len(iou_list),len(type_list),len_label), dtype=bool)
    pred_flag = np.zeros((len(iou_list),len(type_list),len_pred), dtype=bool)
    
    total_matched = np.zeros(( len(iou_list), len(type_list) )) 

    # iou_mat_for_types = { type_i: np.zeros( (len(iou_list),len_label, len_pred), dtype=bool) for type_i in type_list }
    iou_mat_for_types = { type_i: np.zeros( (len_label, len_pred)) for type_i in type_list }
    for i_label in range(len_label):
        for i_pred in range(len_pred):
            # ious = cal_cell_iou( label[i_label], pred[i_pred], type_list )
            ious = bbox_iou_eval( label[i_label], pred[i_pred], type_list )
            # print(ious)
            for type_one,iou_result in ious.items():
                iou_mat_for_types[type_one][ i_label, i_pred ] = iou_result
    
    for type_i in type_list:
        iou_mat = iou_mat_for_types[type_i]
        for iou_t in iou_list:
            for i_label in range(len_label):
                for i_pred in range(len_pred):
                    if (label_flag[iou2id[ iou_t ], type2id[ type_i],i_label] == False
                        and pred_flag[iou2id[ iou_t ], type2id[ type_i] ,i_pred] == False
                        and iou_mat[i_label,i_pred] >= iou_t):
                            label_flag[iou2id[ iou_t ], type2id[ type_i],i_label] = True
                            pred_flag[iou2id[ iou_t ], type2id[ type_i] ,i_pred] = True
                            total_matched[ iou2id[iou_t], type2id[ type_i] ]+= 1
    # print(total_matched)
    
    p_mat = total_matched/len_pred if len_pred!=0 else np.zeros(( len(iou_list), len(type_list) ))
    r_mat = total_matched/len_label if len_label!=0 else np.zeros(( len(iou_list), len(type_list) ))

    return p_mat, r_mat, total_matched




def cal_f1(label, pred):
    label = json2Relations(label, splitted_content=True)
    pred = json2Relations(pred, splitted_content=True)
    precision, recall = eval_relations(gt=[label], res=[pred], cmp_blank=False)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    return [precision, recall, f1]

def cal_f1_cell_iou(label, pred,iou, type_list):
    precision, recall, total_matched = eval_cell_iou(label = label, pred= pred, iou_list=iou, type_list=type_list)

    f1 = (2.0 * precision * recall / (precision + recall+1e-8))
    return [precision, recall, f1, total_matched]




def single_process(labels, preds):
    scores = dict()
    for key in (labels.keys()):
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1(label, pred)
        scores[key] = score
    return scores

def single_process_cell_iou(labels, preds, iou, type_list):
    scores = dict()
    for key in (labels.keys()):
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1_cell_iou(label, pred, iou ,type_list)
        scores[key] = score
    return scores

def _worker(labels, preds,  keys, result_queue):
    for key in keys:
        label = labels.get(key, '')
        pred = preds.get(key, '')
        score = cal_f1(label, pred)
        result_queue.put((key, score))

def _worker_cell_iou(labels, preds,  keys, iou,type_list,result_queue):
    for key in keys:
        pred = preds.get(key, '')
        label = labels.get(key, '')
        score = cal_f1_cell_iou(label, pred, iou, type_list)
        # print(score)
        result_queue.put( (key, score) )
        


def multi_process(labels, preds, num_workers):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(labels.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker,
            args=(
                labels,
                preds,
                keys[worker_idx::num_workers],
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    scores = dict()
    tq = tqdm.tqdm(total=len(keys))
    for _ in range(len(keys)):
        key, val = result_queue.get()
        scores[key] = val
        P, R, F1 = (100 * np.array(list(scores.values()))).mean(0).tolist()
        tq.set_description('P: %.2f, R: %.2f, F1: %.2f' % (P, R, F1), False)
        tq.update()
    
    return scores

def multi_process_cell_iou(labels, preds, num_workers, iou_list=[0.6], type_list=["segmentation"]):
    import multiprocessing
    manager = multiprocessing.Manager()
    result_queue = manager.Queue()
    keys = list(labels.keys())
    workers = list()
    for worker_idx in range(num_workers):
        worker = multiprocessing.Process(
            target=_worker_cell_iou,
            args=(
                labels,
                preds,
                keys[worker_idx::num_workers],
                iou_list ,
                type_list,
                result_queue
            )
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    scores = dict()
    low_f1_paths = list()
    tq = tqdm.tqdm(total=len(keys))
    for _ in range(len(keys)):
        key, val = result_queue.get()
        # print(key,val)
        scores[key] = val
        P_mat, R_mat, F1_mat, total_matched = (np.array(list(scores.values()))).mean(0)
        if np.mean(F1_mat)<0.6:
            low_f1_paths.append(key)
        
        description_all = str()
        for i_type, type_one in enumerate( type_list ):
            for i_iou, iou_t in enumerate( iou_list):
                P = P_mat[i_iou, i_type]
                R = R_mat[i_iou, i_type]
                F1 = F1_mat[i_iou, i_type]
                
                description_all+= 'P_%s_%s: %.2f, R_%s_%s: %.2f, F1_%s_%s: %.2f' % (type_one,str(iou_t),P,type_one,str(iou_t), R,type_one,str(iou_t), F1)
            
        tq.set_description(description_all, False)
        tq.update()
    
    return scores,low_f1_paths



def evaluate_f1(labels, preds, num_workers=0):
    preds = {idx: pred for idx, pred in enumerate(preds)}
    labels = {idx: label for idx, label in enumerate(labels)}
    if num_workers == 0:
        scores = single_process(labels, preds)
    else:
        scores = multi_process(labels, preds, num_workers)
    sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores = [scores[idx] for idx in sorted_idx]
    return scores

def evaluate_f1_cell_iou(labels, preds, iou_list=[0.5], type_list = ["segmentation"],num_workers=0):

    if num_workers == 0:
        scores = single_process_cell_iou(labels, preds,iou_list, type_list)
    else:
        scores,low_f1_paths = multi_process_cell_iou(labels, preds, num_workers,iou_list, type_list)
    # sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores_only = [scores[idx] for idx in preds.keys()]
    return scores_only, scores, low_f1_paths

def evaluate_f1_grid_iou(labels, preds, iou_list=[0.5], type_list = ["segmentation"],num_workers=0):
    # preds = {pred["image_path"]: pred["cells"] for idx, pred in enumerate(preds)}
    # labels = {label["image_path"]: label["cells"] for idx, label in enumerate(labels)}
    # preds_dict = dict()
    # labels_dict = dict()
    # for label,pred in zip(labels,preds):
    #     labels_dict[ label["image_path"]] = label["cells"]
    #     preds_dict[ label["image_path"] ] = pred["cells"]

    if num_workers == 0:
        scores = single_process_cell_iou(labels, preds,iou_list, type_list)
    else:
        scores,low_f1_paths = multi_process_cell_iou(labels, preds, num_workers,iou_list, type_list)
    # sorted_idx = sorted(list(range(len(list(scores)))), key=lambda idx: list(scores.keys())[idx])
    scores_only = [scores[idx] for idx in preds.keys()]
    return scores_only, scores, low_f1_paths