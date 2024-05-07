import sys, json, tqdm, os
import numpy as np
# sys.path.append('./')
# sys.path.append('../')

from utils import  TEDSMetric, all_gather




def cal(total_pred_htmls, total_label_htmls):
    # total_pred_htmls  = [item.replace(r'<html><body><table><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table></body></html>', r'</tbody></table></body></html>') for item in total_pred_htmls]
    #total_label_htmls = [item.replace(r'<html><body><table><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table></body></html>', r'</tbody></table></body></html>') for item in total_label_htmls]
    teds_metric = TEDSMetric(num_workers=20, structure_only=True)
    teds_info = teds_metric(total_pred_htmls, total_label_htmls)
    
    teds = (sum(teds_info), len(teds_info))
    teds = [sum(item) for item in zip(*all_gather(teds))]
    teds = teds[0]/teds[1]

    total_correct = [(np.array(teds_info)==1).sum()]
    total_correct = all_gather(total_correct)
    Acc = np.array(total_correct).sum() / len(total_label_htmls)
    
    return teds, Acc

num_error = 0
total_pred_htmls, total_label_htmls = [], []
lt_dir = '/train13/cv5/hhzhu2/2022/TSR/dataset/show/yinqing_result/cut_img/wireless/'
# lt_dir = '/train13/cv5/hhzhu2/2022/TSR/dataset/show/yinqing_result/cut_img/wired/'
gt_dir = '/work1/cv5/hhzhu2/2022/TSR/dataset/show/test_wrieless_img_gt/'
# gt_dir = '/work1/cv5/hhzhu2/2022/TSR/dataset/show/test_wired_img_gt/'
with open('/work1/cv5/hhzhu2/2022/TSR/dataset/show/jingping/shaoxian.json', 'r') as f:
# with open('/work1/cv5/hhzhu2/2022/TSR/dataset/show/jingping/youxian.json', 'r') as f:
    test_dict = json.load(f)
    for key, value in tqdm.tqdm(test_dict.items()):
        gt_path = gt_dir + key[:-4]+'.txt'
        lt_path = lt_dir + key[:-4]+'.html'
        if not os.path.exists(lt_path):
            lt = ['<html><body><table><thead></thead><tbody></tbody></table></body></html>']
            num_error += 1
        else:
            lt = [line.strip('\n') for line in open(lt_path, 'r').readlines()]
        gt = [line.strip('\n') for line in open(gt_path, 'r').readlines()]
        
      
        total_pred_htmls.append(lt[0])
        total_label_htmls.append(gt[0])
       


    
print (num_error)
print (cal(total_pred_htmls, total_label_htmls))


