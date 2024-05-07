import sys, json, tqdm
import numpy as np
# sys.path.append('./')
# sys.path.append('../')

from utils import  TEDSMetric, all_gather




def cal(total_pred_htmls, total_label_htmls):
    total_pred_htmls  = [item.replace(r'<html><body><table><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table></body></html>', r'</tbody></table></body></html>') for item in total_pred_htmls]
    total_label_htmls  = [item.replace(r'<html><body><table><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table></body></html>', r'</tbody></table></body></html>') for item in total_label_htmls]
    teds_metric = TEDSMetric(num_workers=20, structure_only=True)
    teds_info = teds_metric(total_pred_htmls, total_label_htmls)
    
    teds = (sum(teds_info), len(teds_info))
    teds = [sum(item) for item in zip(*all_gather(teds))]
    teds = teds[0]/teds[1]

    total_correct = [(np.array(teds_info)==1).sum()]
    total_correct = all_gather(total_correct)
    Acc = np.array(total_correct).sum() / len(total_label_htmls)
    
    return teds, Acc


