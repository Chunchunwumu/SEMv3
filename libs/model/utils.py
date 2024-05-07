from mmcv.cnn.bricks import padding
import torch
import torch.nn.functional as F
from mmcv.ops import batched_nms
import numpy as np
import cv2



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
        for row_ in rs_yc: # 额外的地方补成零，需要将这个实例插入
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

def cal_token_acc(preds, labels, masks):
    B, N, _, _ = preds.shape
    preds = torch.argmax( preds, dim=1) #BHW
    preds = (preds ) # (B, N, H*W)
    labels = (labels ) # (B, N, H*W)
    correct_num = ((preds == labels)*masks).sum()
    # print(preds==labels)
    total_num = masks.sum()
    return correct_num / (total_num + 1e-5)


def cal_token_cm(preds, labels, masks):
    B, N, _, _ = preds.shape
    preds = torch.argmax( preds, dim=1) #BHW
    preds = (preds ) # (B,  H*W)
    labels = (labels ) # (B,  H*W)
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.imshow(preds.detach().cpu()[0],vmin=0,vmax=3)
    # plt.colorbar()
    # plt.subplot(2,1,2)
    # plt.imshow((labels*masks).detach().cpu()[0],vmin=0,vmax=3)
    # plt.colorbar()
    # plt.savefig("./cm_cal.png")
    # plt.close()
    cm = np.zeros( (4,4) )
    for i in range(4):
        for j in range(4):
            pred_j = (preds==j)
            label_i = (labels==i)
            correct_num = (( pred_j == label_i)*masks*label_i).sum()
            # print((preds==j),(labels==i) )
            cm[i,j]=correct_num

    return cm


def cal_merge_acc(preds, labels, masks):
    '''calculate the accuracy of merger
    preds: (B, N, H, W)
    labels: (B, N, H, W)
    masks: (B, H, W)
    '''
    B, N, _, _ = preds.shape
    preds = (preds * masks[:, None]).reshape(B, N, -1) # (B, N, H*W)
    labels = (labels * masks[:, None]).reshape(B, N, -1) # (B, N, H*W)
    correct_num = ((preds == labels).min(-1)[0].float() * labels.max(-1)[0]).sum()
    total_num = labels.max(-1)[0].sum()
    return correct_num / (total_num + 1e-5)


def poly2bbox(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x1,y1,x2,y2,x3,y3,x4,y4]
        
    Returns:
        obbs (torch.Tensor): [x1,y1,x2,y2]
    """

    polys = polys.reshape(-1, 4, 2) # (N, 4, 2)
    x1 = polys[:, :, 0].min(-1)[0] # (N)
    y1 = polys[:, :, 1].min(-1)[0] # (N)
    x2 = polys[:, :, 0].max(-1)[0] # (N)
    y2 = polys[:, :, 1].max(-1)[0] # (N)
    polys = torch.stack([x1, y1, x2, y2], dim=-1) # (N, 4)
    return polys


def poly2obb(polys):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (torch.Tensor): [x_lt,y_lt,x_rt,y_rt,x_rb,y_rb,x_lb,y_lb]
        
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    w1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    w2 = torch.sqrt(
        torch.pow(pt4[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt4[..., 1] - pt3[..., 1], 2))
    h1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt4[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt4[..., 1], 2))
    h2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))

    edge1 = (w1 + w2) / 2
    edge2 = (h1 + h2) / 2
    
    angles_1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles_2 = torch.atan2((pt3[..., 1] - pt4[..., 1]),
                          (pt3[..., 0] - pt4[..., 0]))

    angles = (angles_1 + angles_2) / 2

    angles = (angles + np.pi / 2) % np.pi - np.pi / 2
    x_ctr = (pt1[..., 0] + pt2[..., 0] + pt3[..., 0] + pt4[..., 0]) / 4.0
    y_ctr = (pt1[..., 1] + pt2[..., 1] + pt3[..., 1] + pt4[..., 1]) / 4.0
    width = edge1
    height = edge2

    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def obb2poly(obbs):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        
    Returns:
        polys (torch.Tensor): [x_lt,y_lt,x_rt,y_rt,x_rb,y_rb,x_lb,y_lb]
    """
    x1 = torch.cos(obbs[:, 4]) * (-obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 0]
    x2 = torch.cos(obbs[:, 4]) * (obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 0]
    x3 = torch.cos(obbs[:, 4]) * (-obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 0]
    x4 = torch.cos(obbs[:, 4]) * (obbs[:, 2]/2) - torch.sin(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 0]
    y1 = torch.sin(obbs[:, 4]) * (-obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 1]
    y2 = torch.sin(obbs[:, 4]) * (obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (-obbs[:, 3]/2) + obbs[:, 1]
    y3 = torch.sin(obbs[:, 4]) * (-obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 1]
    y4 = torch.sin(obbs[:, 4]) * (obbs[:, 2]/2) + torch.cos(obbs[:, 4]) * (obbs[:, 3]/2) + obbs[:, 1]
    polys = torch.stack([x1,y1,x2,y2,x4,y4,x3,y3], dim=-1)
    return polys

def layout_update_new_token(layouts, insert_rows, insert_cols,row_center_points, col_center_points):
    layouts_list = []
    for layout, insert_row, insert_col, row_p, col_p in zip(layouts, insert_rows, insert_cols, row_center_points, col_center_points):
        layout = layout[:len(row_p)-1, :len(col_p)-1]
        for row in insert_row[1]:
            layout = torch.cat((layout[:row], layout[row-1:]),dim=0)
        for col in insert_col[1]:
            layout = torch.cat((layout[:,:col], layout[:, col-1:]),dim=1)
        layouts_list.append(layout)
    token_map_list = layout_to_token_id(layouts_list)
    return aligned_layouts(token_map_list)

def layout_to_token_id(layouts_list):
    batch_size = len(layouts_list)
    token_map_list = list()
    for batch_idx in range(batch_size):
        token_map = torch.zeros_like(layouts_list[batch_idx])
        cell_ids = torch.unique( layouts_list[batch_idx] )
        for ids in cell_ids:
            indexs = torch.nonzero(layouts_list[batch_idx]==ids, )
            begin_index_y, begin_index_x = indexs[0]
            token_map[begin_index_y, begin_index_x] = 0 # c
            for y,x in indexs[1:]:
                if y == begin_index_y:
                    token_map[y,x]=1 # l
                elif x== begin_index_x:
                    token_map[y,x]=2 # u
                else:
                    token_map[y,x]=3 # x
        token_map_list.append(token_map)
    return token_map_list

      

def layout_update(layouts, insert_rows, insert_cols,row_center_points, col_center_points):
    layouts_list = []
    for layout, insert_row, insert_col, row_p, col_p in zip(layouts, insert_rows, insert_cols, row_center_points, col_center_points):
        layout = layout[:len(row_p)-1, :len(col_p)-1]
        for row in insert_row[1]:
            layout = torch.cat((layout[:row], layout[row-1:]),dim=0)
        for col in insert_col[1]:
            layout = torch.cat((layout[:,:col], layout[:, col-1:]),dim=1)
        layouts_list.append(layout)
    return aligned_layouts(layouts_list)

def aligned_layouts(layouts_list):
    batch_size = len(layouts_list)
    
    max_row_nums = max([l.shape[0] for l in layouts_list])
    max_col_nums = max([l.shape[1] for l in layouts_list])

    aligned_layouts = list()
    for batch_idx in range(batch_size):
        num_rows_pi = layouts_list[batch_idx].shape[0]
        num_cols_pi = layouts_list[batch_idx].shape[1]
        layouts_pi = layouts_list[batch_idx]
        aligned_layouts_pi = F.pad(
            layouts_pi,
            (0, max_col_nums-num_cols_pi, 0, max_row_nums-num_rows_pi),
            mode='constant',
            value=-100
        )
        aligned_layouts.append(aligned_layouts_pi)
    aligned_layouts = torch.stack(aligned_layouts, dim=0)
    return aligned_layouts