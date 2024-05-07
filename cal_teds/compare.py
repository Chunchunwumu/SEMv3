import glob, cv2, tqdm
import os,shutil

from list_record_cache import ListRecordLoader


def _worker(path_list, teds_list, teds_ori_dict, wireless_dict):
    for path, teds_new in tqdm.tqdm(zip(path_list, teds_list)):
        
        path_t = wireless_dict[path]
        path_t = path_t.replace('/ps6/cv2/hhzhu2/2022/TSR/dataset/01豬玖ｯ暮寔', '01_test')
        teds_ori = teds_ori_dict[path_t]

        path = os.path.splitext(path)[0]
        if teds_new >= teds_ori:
            name = '{}_{}_{}_{}'.format(round(teds_new-teds_ori, 2), round(teds_new, 2), round(teds_ori, 2), path).replace('ps6.cv2.hhzhu2.2022.TSR.dataset.', '')
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'.jpg'), os.path.join(save_path+'good', name+'.jpg'))
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'.html'), os.path.join(save_path+'good', name+'_new.html'))
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'_gt.html'), os.path.join(save_path+'good', name+'_gt.html'))

            shutil.copy(os.path.join(os.path.dirname(teds_path_ori)+'/img_html/', 'table_data_0617#'+os.path.splitext(path_t)[0].replace('/', '#')+'.html'), os.path.join(save_path+'good', name+'_old.html'))
        else:
            name = '{}_{}_{}_{}'.format(round(teds_ori-teds_new, 2), round(teds_new, 2), round(teds_ori, 2), path).replace('ps6.cv2.hhzhu2.2022.TSR.dataset.', '')
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'.jpg'), os.path.join(save_path+'bad', name+'.jpg'))
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'.html'), os.path.join(save_path+'bad', name+'_new.html'))
            shutil.copy(os.path.join(os.path.dirname(teds_path_new), path+'_gt.html'), os.path.join(save_path+'bad', name+'_gt.html'))

            shutil.copy(os.path.join(os.path.dirname(teds_path_ori)+'/img_html/', 'table_data_0617#'+os.path.splitext(path_t)[0].replace('/', '#')+'.html'), os.path.join(save_path+'bad', name+'_old.html'))

            

def multi_process(teds_new_dict, teds_ori_dict, wireless_dict, num_workers=1):
    import multiprocessing
   
    path_list = list(teds_new_dict.keys())
    teds_list = list(teds_new_dict.values())
  
    if num_workers == 1:
        _worker(path_list, teds_list, teds_ori_dict, wireless_dict)
    else:
        workers = list()
        for worker_idx in range(num_workers):
            worker = multiprocessing.Process(
                target=_worker,
                args=(
                    path_list[worker_idx::num_workers],
                    teds_list[worker_idx::num_workers],
                    teds_ori_dict,
                    wireless_dict
                )
            )
            worker.daemon = True
            worker.start()
            workers.append(worker)
    
   
teds_path_ori = '/work1/cv5/hhzhu2/2022/TSR/dataset/show/test_result/baseline_ori/result_teds.txt'
teds_path_new = '/work1/cv5/hhzhu2/2022/TSR/dataset/show/test_result/SEMv2_v1.1.3.4/result_teds.txt'
save_path = '/work1/cv5/hhzhu2/2022/TSR/dataset/show/test_result/SEMv2_v1.1.3.4_baseline_ori_compare/'
if os.path.exists(save_path):
    shutil.rmtree(save_path)
os.makedirs(save_path+'good')
os.makedirs(save_path+'bad')

teds_ori_dict = {}
with open(teds_path_ori, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').split()
        name = ' '.join(line[:-1]).replace('/ps3/cv8/dwzeng/original_data/table_data/table_data_2022/table_data_0617/', '')[:-1]
        teds_ori_dict[name] = float(line[-1])

teds_new_dict = {}
with open(teds_path_new, 'r') as f:
    for line in f.readlines():
        line = line.strip('\n').split('\t')
        name = os.path.basename(line[0]).replace('/ps6/cv2/hhzhu2/2022/TSR/dataset/', '')
        teds_new_dict[name] = float(line[-1])


lrc_path = "/work1/cv5/hhzhu2/2022/TSR/dataset/lrc_file/test_wireless_table_update20220725_compare.lrc"

loader = ListRecordLoader(lrc_path)

wireless_dict = {}
for idx in range(len(loader)):
    info = loader.get_record(idx)
    wireless_dict['{}.jpg'.format(os.path.splitext(os.path.basename(info['image_path']))[0])] = info['orgin_image_path']

multi_process(teds_new_dict, teds_ori_dict, wireless_dict, num_workers=10)
