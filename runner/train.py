import sys
from traceback import print_exc
import numpy as np
sys.path.append('./')
sys.path.append('../')
import os
import torch
import cv2
from collections import defaultdict
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm.scheduler
from libs.configs import cfg, setup_config
from libs.model import build_model
from libs.data import create_train_dataloader, create_valid_dataloader
from libs.utils import logger
from libs.utils.counter import Counter
from libs.utils.utils import cal_mean_lr
from libs.utils.utils import is_simple_table
from libs.utils.format_translate import table_to_html, format_html
from libs.utils.checkpoint import load_checkpoint, save_checkpoint
from libs.utils.time_counter import TimeCounter
from libs.utils.comm import distributed, synchronize,all_gather
from libs.utils.model_synchronizer import ModelSynchronizer
from libs.utils.cal_f1 import evaluate_f1_cell_iou, pred_result_to_table, gt_format_v1,gt_format_delta, pred_result_to_table_new_token_iflytab, table_to_relations, evaluate_f1, gt_format
from libs.utils.metric import  TEDSMetric
import traceback
import time
import math
import shutil
import torch.nn.functional as F
import tqdm
from libs.utils.eval import eval

metrics_name = ['cell_iou_f1']
best_metrics = [0.0]
def get_exist_path(path):
    os.makedirs(path,exist_ok=True)
    return path

def init():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default='default')
    parser.add_argument("--work_dir", type=str, default=None)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--version_name",type=str, default="v1")
    args = parser.parse_args()
    
    setup_config(args.cfg)
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir+"_"+cfg.tips
    
    get_exist_path(cfg.work_dir)
    # print(os.path.join( cfg.work_dir, args.cfg+".py"))
    if not os.path.isfile(os.path.join( cfg.work_dir, args.cfg+".py")):
        shutil.copyfile(os.path.join(f"./libs/configs",args.cfg+".py"), os.path.join( cfg.work_dir, args.cfg+".py" ))
    
    
    os.environ['LOCAL_RANK'] = str(args.local_rank)
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    logger.setup_logger('Line Detect Model', cfg.work_dir, 'train.log')
    logger.info('Use config: %s' % args.cfg)


def valid(cfg, dataloader, model):
    model.eval()
    total_label_relations = list()
    total_pred_relations = list()
    total_relations_metric = list()
    total_label_htmls = list()
    total_pred_htmls = list()
    total_tables_type = list()
    
    for it, data_batch in enumerate(dataloader):
        images = data_batch['images'].to(cfg.device)
        images_size = data_batch['images_size'].to(cfg.device)
        tables = data_batch['tables']
        for table_idx in range(len(tables)):
            scale_w, scale_h = tables[table_idx]['scale_rate']
            assert scale_w == scale_h
            tables[table_idx]['line_polys'] = [ np.array(item/scale_w).astype(np.int64) for item in  tables[table_idx]['line_polys']]
            
        
        # pred
        _, row_start_bboxes, row_center_points, row_segm_logits, \
            col_start_bboxes, col_center_points, col_segm_logits, \
                mg_logits, num_rows, num_cols, gird_bboxes = model(images, images_size)

        pred_tables = [
            pred_result_to_table_new_token_iflytab(tables[batch_idx],
                (row_center_points[batch_idx], row_segm_logits[batch_idx], \
                    col_center_points[batch_idx], col_segm_logits[batch_idx], \
                        mg_logits[batch_idx], num_rows[batch_idx], num_cols[batch_idx], gird_bboxes[batch_idx])
            ) \
            for batch_idx in range(len(images_size))
        ] # NOTE: set content as 'no content'
        
        pred_relations = [table_to_relations(table) for table in pred_tables]
        total_pred_relations.extend(pred_relations)
        tables = [gt_format_delta(tables[b_i]) for b_i in range(len(tables))]
        label_relations = [table_to_relations(table) for table in tables]
        total_label_relations.extend(label_relations)

        pred_htmls = [table_to_html(table) for table in pred_tables]
        total_pred_htmls.extend([format_html(item) for item in pred_htmls])

        label_htmls = [table_to_html(table) for table in tables]
        total_label_htmls.extend([format_html(item) for item in label_htmls])

        tables_type = ['Simple' if is_simple_table(table) else 'Complex' for table in tables]
        total_tables_type.extend(tables_type)

    # new evaluate P, R, F1, Acc
    total_relations_metric = evaluate_f1(total_label_relations, total_pred_relations, num_workers=20)
    total_relations_metric_p  = [sum(np.array(total_relations_metric)[:, 0].tolist())]
    total_relations_metric_r  = [sum(np.array(total_relations_metric)[:, 1].tolist())]
    total_len = [float(len(total_relations_metric))]

    total_relations_metric_p = all_gather(total_relations_metric_p)
    total_relations_metric_r = all_gather(total_relations_metric_r)
    total_len = all_gather(total_len)
    P = np.array(total_relations_metric_p).sum() / np.array(total_len).sum()
    R = np.array(total_relations_metric_r).sum() / np.array(total_len).sum()
    F1 = 2 * P * R / (P + R+1e-5)
    logger.info('[Valid] Total Type Mertric: Precision: %s, Recall: %s, F1-Score: %s' % (P, R, F1))

    # evaluate TEDS
    teds_metric = TEDSMetric(num_workers=40, structure_only=True)
    teds_info = teds_metric(total_pred_htmls, total_label_htmls)
    
    teds = (sum(teds_info), len(teds_info))
    teds = [sum(item) for item in zip(*all_gather(teds))]
    teds = teds[0]/teds[1]

    total_correct = [(np.array(teds_info)==1).sum()]
    total_correct = all_gather(total_correct)
    Acc = np.array(total_correct).sum() / np.array(total_len).sum()

    typed_teds = defaultdict(list)
    for table_type, teds_info_ps in zip(total_tables_type, teds_info):
        typed_teds[table_type].append(teds_info_ps)
    typed_teds = {key: (sum(val), len(val)) for key, val in typed_teds.items()}

    total_typed_teds = dict()
    for typed_teds_pw in all_gather(typed_teds):
        for key, val in typed_teds_pw.items():
            if key not in total_typed_teds:
                total_typed_teds[key] = val
            else:
                exist_val = total_typed_teds[key]
                total_typed_teds[key] = (val[0] + exist_val[0], val[1] + exist_val[1])
    typed_teds = {key: val[0]/val[1] for key, val in total_typed_teds.items()}

    logger.info('[Valid] Teds: %s, Acc: %s' % (teds, Acc))
    for key, val in typed_teds.items():
        logger.info('    %s Teds: %s' % (key, val))
    
    return (F1, )
 


def train(cfg, epoch, dataloader, model, optimizer, scheduler, time_counter, synchronizer=None, epoch_item=0):
    model.train()
    counter = Counter(cache_nums=cfg.log_sep)
    for it, data_batch in enumerate(dataloader):
        try:
            images = data_batch['images'].to(cfg.device)
            images_size = data_batch['images_size'].to(cfg.device)
            row_start_bboxes = [item.to(cfg.device) for item in data_batch['row_start_bboxes']]
            col_start_bboxes = [item.to(cfg.device) for item in data_batch['col_start_bboxes']]
            col_deltas = data_batch["col_deltas"]
            row_deltas = data_batch["row_deltas"]
            layouts = data_batch['layouts'].to(cfg.device)
            optimizer.zero_grad()
            result_info, row_start_bboxes, row_center_points, row_deltas, \
            col_start_bboxes, col_center_points, col_deltas, mg_logits, num_rows, num_cols, table_gird_bboxes = model(
                images, images_size, 
                row_start_bboxes,
                col_start_bboxes,
                layouts, epoch_item,col_deltas=col_deltas,
                row_deltas=row_deltas
            )
            counter.update(result_info)
            loss = sum([val for key, val in result_info.items() if 'loss' in key])
            loss.backward()
            optimizer.step()
            scheduler.step()
            counter.update(result_info)
        except RuntimeError as E:
            if 'out of memory' in str(E):
                logger.info('CUDA Out Of Memory')
                traceback.print_exc()

                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(E))
                traceback.print_exc()
        except:
            logger.info("error")
            traceback.print_exc()

        if it % cfg.log_sep == 0:
            logger.info(
                '[Train][Epoch %03d Iter %04d][Memory: %.0f][Mean LR: %f][Left: %s] %s' % \
                (
                    epoch,
                    it,
                    torch.cuda.max_memory_allocated()/1024/1024,
                    cal_mean_lr(optimizer),
                    time_counter.step(epoch, it + 1),
                    counter.format_mean(sync=False)
                )
            )
        
        if synchronizer is not None:
            synchronizer()
    if synchronizer is not None:
        synchronizer(final_align=True)


def build_optimizer(cfg, model):
    params = list()
    for _, value in model.named_parameters():
        if not value.requires_grad:
            continue
        
        lr = cfg.base_lr
        weight_decay = cfg.weight_decay
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    
    optimizer = torch.optim.Adam(params, cfg.base_lr)
    return optimizer

def draw_segmentation(image, segmentation, color):
        for contour in segmentation:
            color = (int(np.random.randint(10,255,1)),int(np.random.randint(10,255,1)),int(np.random.randint(10,255,1)))
            contour = np.array(contour, dtype=np.int32).reshape((-1,2))
            image = cv2.polylines(image, [contour], True, color)
        return image

def build_scheduler(cfg, optimizer, epoch_iters, start_epoch=0):
    scheduler = CosineAnnealingLR(
        optimizer=optimizer,
        T_max=cfg.num_epochs * epoch_iters,
        eta_min=cfg.min_lr,
        last_epoch=-1 if start_epoch == 0 else start_epoch * epoch_iters
    )
    return scheduler


def main():
    init()

    np.random.seed(0)
    scale_bucket =  0.8 + (1.2-0.8)*np.random.random((cfg.num_epochs,1000))
    
    train_dataloader = create_train_dataloader( 
        cfg.train_lrcs_path,
        cfg.train_num_workers,
        cfg.train_max_batch_size,
        cfg.train_max_pixel_nums,
        cfg.train_max_row_nums,
        cfg.train_max_col_nums,
        cfg.train_bucket_seps,
        cfg.max_img_size,
        cfg.height_norm,
        cfg.train_rota,
        0,
        scale_bucket
    )

    logger.info(
        'Train dataset have %d samples, %d batchs' % \
            (
                len(train_dataloader.dataset),
                len(train_dataloader.batch_sampler)
            )
    )

    valid_dataloader = create_valid_dataloader(
        cfg.valid_lrc_path,
        cfg.valid_num_workers,
        cfg.valid_batch_size,
        cfg.max_img_size,
        cfg.height_norm
    )

    logger.info(
        'Valid dataset have %d samples, %d batchs with batch_size=%d' % \
            (
                len(valid_dataloader.dataset),
                len(valid_dataloader.batch_sampler),
                valid_dataloader.batch_size
            )
    )

    model = build_model(cfg)
    model.to(cfg.device)

    if distributed():
        print(os.environ["WORLD_SIZE"])
        synchronizer = ModelSynchronizer(model, cfg.sync_rate)
    else:
        synchronizer = None

    epoch_iters = len(train_dataloader.batch_sampler)
    optimizer = build_optimizer(cfg, model)

    global metrics_name
    global best_metrics
    start_epoch = 0
    if cfg.resume_path ==None:

        resume_path =  os.path.join(cfg.work_dir, 'latest_model.pth')
    else:
        resume_path =  cfg.resume_path
    if os.path.exists(resume_path):
        best_metrics, start_epoch = load_checkpoint(resume_path, model, optimizer)
        if len(best_metrics)==1:
            best_metrics.append(0)
        start_epoch += 1
        logger.info('resume from: %s' % resume_path)
    elif cfg.train_checkpoint is not None:
        load_checkpoint(cfg.train_checkpoint, model)
        logger.info('load checkpoint from: %s' % cfg.train_checkpoint)
    
    scheduler = build_scheduler(cfg, optimizer, epoch_iters, start_epoch)
    
    time_counter = TimeCounter(start_epoch, cfg.num_epochs, epoch_iters)
    time_counter.reset()
    
    for epoch in range(start_epoch, cfg.num_epochs):
        np.random.seed(epoch)
        scale_bucket =  0.8 + (1.2-0.8)*np.random.random((cfg.num_epochs,1000))
    
        train_dataloader = create_train_dataloader( 
            cfg.train_lrcs_path,
            cfg.train_num_workers,
            cfg.train_max_batch_size,
            cfg.train_max_pixel_nums,
            cfg.train_max_row_nums,
            cfg.train_max_col_nums,
            cfg.train_bucket_seps,
            cfg.max_img_size,
            cfg.height_norm,
            cfg.valid_rota,
            epoch,
            scale_bucket
        )

        if hasattr(train_dataloader.batch_sampler, 'set_epoch'):
            train_dataloader.batch_sampler.set_epoch(epoch)

        logger.info(
            'Train dataset have %d samples, %d batchs' % \
                (
                    len(train_dataloader.dataset),
                    len(train_dataloader.batch_sampler)
                )
        )

        train(cfg, epoch, train_dataloader, model, optimizer, scheduler, time_counter, synchronizer, epoch)

        if epoch >= cfg.start_eval and epoch % cfg.eval_epochs == 0:
            with torch.no_grad():
                # metrics = valid(cfg, valid_dataloader, model)
                metrics = valid(cfg, valid_dataloader, model)

            for metric_idx in range(len(metrics_name)):
                if metrics[metric_idx] > best_metrics[metric_idx]:
                    best_metrics[metric_idx] = metrics[metric_idx]
                    save_checkpoint(os.path.join(cfg.work_dir, 'best_%s_model.pth' % metrics_name[metric_idx]), model, optimizer, best_metrics, epoch)
                    logger.info('Save current model as best_%s_model' % metrics_name[metric_idx])

        save_checkpoint(os.path.join(cfg.work_dir, 'latest_model.pth'), model, optimizer, best_metrics, epoch)


if __name__ == '__main__':
    main()
