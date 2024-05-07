import cv2
import mmcv
import numpy as np
from .utils import Resize_iflytab, PhotoMetricDistortion
from torchvision.transforms import functional as F
import torch
import matplotlib.pyplot as plt


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, *data):
        for transform in self.transforms:
            data = transform(*data)
        return data

class CallResizeImage:
    def __init__(self, **kwargs):
        self.operation = Resize_iflytab(**kwargs)
    
    def __call__(self, image, table):
        table.update(img=image)
        table = self.operation(table)
        image = table['img']
        return image, table


class CallImageDistortion:
    def __init__(self, **kwargs):
        self.operation = PhotoMetricDistortion(**kwargs)
    
    def __call__(self, image, table):
        # image = self.operation(image).astype('uint8')
        pass
        return image, table



class CallRowColStartBox_v0:
    # 不调整宽度
    def __call__(self, image, table):
        table.update(img=image)
        table.update(img_pad=image)
        image_h, image_w = image.shape[0:2]
        if 'row_start_center_bboxes' in table and 'col_start_center_bboxes' in table:
            row_start_bboxes = np.array(table['row_start_center_bboxes'], dtype=np.float32)
            col_start_bboxes = np.array(table['col_start_center_bboxes'], dtype=np.float32)
            row_start_bboxes_np = row_start_bboxes
            col_start_bboxes_np = col_start_bboxes
            row_start_bboxes_center =  (row_start_bboxes_np[:,1] + row_start_bboxes_np[:,3])/2
            col_start_bboxes_center =  (col_start_bboxes_np[:,0] + col_start_bboxes_np[:,2])/2
            row_bbox_h = np.clip(np.abs((row_start_bboxes_np[:,3] - row_start_bboxes_np[:,1] )/2), a_min=4, a_max=5000)
            col_bbox_w = np.clip(np.abs((col_start_bboxes_np[:,2] - col_start_bboxes_np[:,0] )/2), a_min=4, a_max=5000)

            row_start_bboxes_np[:, 1] = np.clip( row_start_bboxes_center-row_bbox_h, 0, image_h - 1 )
            row_start_bboxes_np[:, 3] = np.clip( row_start_bboxes_center+row_bbox_h, 0, image_h - 1 )
            col_start_bboxes_np[:, 0] = np.clip( col_start_bboxes_center-col_bbox_w, 0, image_w - 1 )
            col_start_bboxes_np[:, 2] = np.clip( col_start_bboxes_center+col_bbox_w, 0, image_w - 1 )
        else:
            row_start_bboxes = np.zeros((0, 4), dtype=np.float32)
            col_start_bboxes = np.zeros((0, 4), dtype=np.float32)
        
        table["row_start_bboxes"] = row_start_bboxes
        table["col_start_bboxes"] = col_start_bboxes
        return image, table, row_start_bboxes, col_start_bboxes

class CallImageNormalize:
    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, image, table, row_start_bboxes, col_start_bboxes):
        return image, table, row_start_bboxes, col_start_bboxes


class CallRowColLineMask:
    def __call__(self, image, table, row_start_bboxes, col_start_bboxes):
        if 'row_split_lines' in table and  'col_split_lines' in table:
            image_h, image_w = image.shape[0:2]
            row_deltas = list()
            for segm in table['row_split_lines']:
                # 隶｡邂妖elta
                segm = np.round(np.array(segm).reshape(-1,1,2)).astype('int')

                deltas_temp = np.zeros(image_w)
                point = np.array([0,segm[0][0][1]])
                start_y = point[1]
                for point_now in segm[:,0,:]:
                    slope = (point_now[1] - point[1])/((point_now[0] - point[0])+1e-5)
                    xs = np.arange(point[0], point_now[0])-point[0]
                    deltas = xs*slope + (point[1]-start_y)
                    deltas_temp[point[0]:point_now[0]] = deltas
                    point = point_now
                point_now = np.array([image_w-1,point[1]])
                slope = (point_now[1] - point[1])/((point_now[1] - point[1])+1e-5)
                xs = np.arange(point[0], point_now[0]+1)-point[0]
                deltas = xs*slope + (point[1]-start_y)
                deltas_temp[point[0]:point_now[0]+1] = deltas

                row_deltas.append(deltas_temp)

            col_deltas = list()
            for segm in table['col_split_lines']:
                segm = np.round(np.array(segm).reshape(-1,1,2)).astype('int')

                deltas_temp = np.zeros(image_h)
                point = np.array([segm[0][0][0],0])
                start_x = point[0]
                for point_now in segm[:,0,:]:
                    slope = (point_now[0] - point[0])/((point_now[1] - point[1])+1e-5)
                    ys = np.arange(point[1], point_now[1])-point[1]
                    deltas = ys*slope + (point[0]-start_x)
                    deltas_temp[point[1]:point_now[1]] = deltas
                    point = point_now
                point_now = np.array([point[0],image_h-1])
                slope = (point_now[0] - point[0])/((point_now[1] - point[1])+1e-5)
                ys = np.arange(point[1], point_now[1]+1)-point[1]
                deltas = ys*slope + (point[0]-start_x)
                deltas_temp[point[1]:point_now[1]+1] = deltas

                col_deltas.append(deltas_temp)
 
        else:
            col_deltas = list()
            row_deltas = list()
        col_deltas = np.array(col_deltas)
        row_deltas = np.array(row_deltas)
        # print(row_deltas)

        table["row_deltas"] = row_deltas
        table["col_deltas"] = col_deltas
        if 'row_line_segmentations' in table and  'col_line_segmentations' in table:
            image_h, image_w = image.shape[-2:]

            row_line_masks = list()
            for segm in table['row_line_segmentations']:
                canvas = np.zeros((image_h, image_w))
                segm = np.array(segm).reshape(-1,1,2).astype('int')
                if table['is_wireless']:
                    cv2.fillPoly(canvas, [segm], 1)
                else:
                    cv2.polylines(canvas, [segm], False, 1, 5)
                row_line_masks.append(canvas)
            row_line_masks = np.concatenate([item[None] for item in row_line_masks], axis=0).astype(np.int64)

            col_line_masks = list()
            for segm in table['col_line_segmentations']:
                canvas = np.zeros((image_h, image_w))
                segm = np.array(segm).reshape(-1,1,2).astype('int')
                if table['is_wireless']:
                    cv2.fillPoly(canvas, [segm], 1)
                else:
                    cv2.polylines(canvas, [segm], False, 1, 5)
                col_line_masks.append(canvas)
            col_line_masks = np.concatenate([item[None] for item in col_line_masks], axis=0).astype(np.int64)

        else:
            image_h, image_w = image.shape[-2:] # (3, h, w)
            row_line_masks = np.zeros((0, image_h, image_w), dtype=np.int64)
            col_line_masks = np.zeros((0, image_h, image_w), dtype=np.int64)

        table['row_line_masks'] = row_line_masks
        table['col_line_masks'] = col_line_masks
        return image, table, row_start_bboxes, col_start_bboxes,row_deltas, col_deltas
  
def get_32(a):
    if a%32!=0:
        a = (a//32+1)*32
    return a


class CallRandomRotation_normalize:
    def __init__(self, mean, std, inplace=False,is_rota=True):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.is_rota = is_rota
        
    def __call__( self, image, table, row_start_bboxes, col_start_bboxes, row_deltas, col_deltas):

        h,w = image.shape[0:2]
            
        table["row_start_bboxes"] = row_start_bboxes
        table["col_start_bboxes"] = col_start_bboxes
        table['row_start_center_bboxes'] = row_start_bboxes
        table['col_start_center_bboxes'] = col_start_bboxes
        
        h_32 = get_32(table['image_h'])
        w_32 = get_32(table['image_w'])
        
        
        table['img_pad'] = image
        image_pad = np.pad( image.transpose(2,0,1) , pad_width=( (0,0), (0,h_32- table['image_h'] ),(0, w_32-table['image_w'])), mode='constant' )
        ## image normalize
        image = torch.from_numpy(image_pad).to(torch.float32)
        image = image/255
        image = F.normalize(image, self.mean, self.std, self.inplace)
        
        return image, table, row_start_bboxes, col_start_bboxes, row_deltas, col_deltas
        
        
class CallLayout:
    def __call__(self, image, table, row_start_bboxes, col_start_bboxes, row_deltas, col_deltas):
        if 'layout' in table:
            layout = np.array(table['layout'], dtype=np.int64) # (NumRow, NumCol)
        else:
            layout = np.zeros((0, 0),  dtype=np.int64)
        return image, table, row_start_bboxes, col_start_bboxes,  row_deltas, col_deltas, layout
