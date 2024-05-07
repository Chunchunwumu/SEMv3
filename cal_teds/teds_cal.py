import sys
import numpy as np
# sys.path.append('./')
# sys.path.append('../')

from utils import  TEDSMetric, all_gather




def cal(total_pred_htmls, total_label_htmls):
    total_pred_htmls  = [item.replace(r'<table border="1"><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table>', r'</tbody></table></body></html>') for item in total_pred_htmls]
    total_label_htmls = [item.replace(r'<table border="1"><thead>', r'<html><body><table><thead></thead><tbody>').replace(r'</thead><tbody></tbody></table>', r'</tbody></table></body></html>') for item in total_label_htmls]
    teds_metric = TEDSMetric(num_workers=20, structure_only=True)
    teds_info = teds_metric(total_pred_htmls, total_label_htmls)
    
    teds = (sum(teds_info), len(teds_info))
    teds = [sum(item) for item in zip(*all_gather(teds))]
    teds = teds[0]/teds[1]

    total_correct = [(np.array(teds_info)==1).sum()]
    total_correct = all_gather(total_correct)
    Acc = np.array(total_correct).sum() / len(total_label_htmls)
    
    return teds, Acc



total_pred_htmls = [r'<table border="1"><thead><tr><td></td><td>000</td><td>00000</td><td>002</td></tr><tr><td rowspan="6">006</td><td>004</td><td>003</td><td>005</td></tr><tr><td>007</td><td>008</td><td>009</td></tr><tr><td>012</td><td>010</td><td>011</td></tr><tr><td>013</td><td>014</td><td>015</td></tr><tr><td>017</td><td>016</td><td>018</td></tr><tr><td>019</td><td>020</td><td>021</td></tr></thead><tbody></tbody></table>', r'<table border="1"><thead><tr><td></td><td>000</td><td>00000</td><td>002</td></tr><tr><td rowspan="6">006</td><td>004</td><td>003</td><td>005</td></tr><tr><td>007</td><td>008</td><td>009</td></tr><tr><td>012</td><td>010</td><td>011</td></tr><tr><td>013</td><td>014</td><td>015</td></tr><tr><td>017</td><td>016</td><td>018</td></tr><tr><td>019</td><td>020</td><td>021</td></tr></thead><tbody></tbody></table>']
total_label_htmls = [r'<table border="1"><thead><tr><td></td><td>000</td><td>001</td><td>002</td></tr><tr><td rowspan="5">006</td><td>004</td><td>003</td><td>005</td></tr><tr><td>007</td><td>008</td><td>009</td></tr><tr><td>012</td><td>010</td><td>011</td></tr><tr><td>013</td><td>014</td><td>015</td></tr><tr><td>017</td><td>016</td><td>018</td></tr><tr><td>019</td><td>020</td><td>021</td></tr></thead><tbody></tbody></table>', r'<table border="1"><thead><tr><td></td><td>000</td><td>00000</td><td>002</td></tr><tr><td rowspan="6">006</td><td>004</td><td>003</td><td>005</td></tr><tr><td>007</td><td>008</td><td>009</td></tr><tr><td>012</td><td>010</td><td>011</td></tr><tr><td>013</td><td>014</td><td>015</td></tr><tr><td>017</td><td>016</td><td>018</td></tr><tr><td>019</td><td>020</td><td>021</td></tr></thead><tbody></tbody></table>']
print (cal(total_pred_htmls, total_label_htmls))


