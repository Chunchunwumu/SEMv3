import cv2
import sys
import json

from torch.nn.functional import normalize
sys.path.append('./')
sys.path.append('../')
import os
import tqdm
import shutil
import torch
import numpy as np
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data.transform import CallImageNormalize_infer
from libs.data import create_valid_dataloader
from libs.utils import logger
from libs.utils.visualize import visualize
from libs.utils.checkpoint import load_checkpoint
from libs.utils.comm import synchronize, all_gather
from libs.utils.format_translate import table_to_html, format_html
from libs.data.utils import norm_by_height
from libs.utils.cal_f1 import  pred_result_to_table_new_token_iflytab, table_to_relations, evaluate_f1


def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lrc", type=str, default=None)
    parser.add_argument("--cfg", type=str, default='config_2')
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--work_dir", type=str, default='./')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.lrc is not None:
        cfg.valid_lrc_path = args.lrc
    
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    
    if args.image_dir is not None:
        cfg.image_dir = args.image_dir

    os.environ['LOCAL_RANK'] = str(args.local_rank)

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'valid.log')
    logger.info('Use config: %s' % args.cfg)
    logger.info('Evaluate Dataset: %s' % cfg.valid_lrc_path)


def valid(cfg, image_files, model):
    model.eval()
    
    vis_dir = os.path.join(cfg.work_dir, 'infer_folder')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir)
    vis_dir = os.path.abspath(vis_dir)
    
    image_normalize = CallImageNormalize_infer(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    for it, image_file in enumerate(tqdm.tqdm(image_files)):

        table = dict()
        image = cv2.imread(image_file)
        height = norm_by_height(image)
        height_ratio = 10 / height

        img_sacle = image.shape[:2][::-1]
        scale = (int(img_sacle[0]*height_ratio), int(img_sacle[1]*height_ratio))
       
        image = cv2.resize(image, scale)

        table.update(img=image)
        image_size = image.shape[:2][::-1] # w, h
        image_size = torch.tensor(image_size).long()
        image = image_normalize(image)
        images = image.unsqueeze(0).to(cfg.device)
        images_size = image_size.unsqueeze(0).to(cfg.device)
        tables = [table]
        
        # pred
        _, row_start_bboxes, row_center_points, row_segm_logits, \
            col_start_bboxes, col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, table_gird_bboxes = model(images, images_size)

        pred_tables = [
            pred_result_to_table_new_token_iflytab(tables[batch_idx],
                (row_center_points[batch_idx], row_segm_logits[batch_idx], \
                    col_center_points[batch_idx], col_segm_logits[batch_idx], \
                        mg_logits[batch_idx], num_rows[batch_idx], num_cols[batch_idx], table_gird_bboxes[batch_idx]), is_infer=True
            ) \
            for batch_idx in range(len(images_size))
        ] # NOTE: set content as 'no content'

        pred_htmls = [table_to_html(table) for table in pred_tables]

        path = './'
        name = os.path.join(path, os.path.splitext(os.path.basename(image_file))[0]+'.html')
        with open(name, 'w') as f_w:
            f_w.writelines([format_html(item) for item in pred_htmls][0])

def main():
    init()

    import glob
    
    image_files = ['/train20/intern/permanent/cxqin/TSR/dataset/hhzhu_dataset/dataset/train_wrieless/table_images/ps6.cv2.hhzhu2.2022.TSR.dataset.02训练集.01商务合同-zslj8160.合同、招投标书-zslj4875.合同-5.24-500-zslj463.少线500-zslj463.屏幕拍照500-zslj463.143-zhaotoubiao-shao-pingmupaizhao.jpg.0.jpg']
    logger.info('Inference image files have %d samples' % len(image_files))

    model = build_model(cfg)
    model.cuda()
    
    eval_checkpoint = os.path.join(cfg.infer_checkpoint)
    load_checkpoint(eval_checkpoint, model)
    logger.info('Load checkpoint from: %s' % eval_checkpoint)

    with torch.no_grad():
        valid(cfg, image_files, model)


if __name__ == '__main__':
    main()
