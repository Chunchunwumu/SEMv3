# import glob, cv2
# import os, tqdm
# import pickle
# import threading, cv2
# import numpy as np
# import pandas as pd 

# class ListRecordLoader:
#     OFFSET_LENGTH = 8
#     def __init__(self, load_path):
#         self._sync_lock = threading.Lock()
#         self._size = os.path.getsize(load_path)
#         self._load_path = load_path
#         self._open_file()
#         self._scan_file()

#     def _open_file(self):
#         self._pid = os.getpid()
#         self._cache_file = open(self._load_path, 'rb')

#     def _check_reopen(self):
#         if (self._pid != os.getpid()):
#             self._open_file()

#     def _scan_file(self):
#         record_pos_list = list()
#         pos = 0
#         while True:
#             if pos >= self._size:
#                 break
#             self._cache_file.seek(pos)
#             offset = int().from_bytes(
#                 self._cache_file.read(self.OFFSET_LENGTH),
#                 byteorder='big', signed=False
#             )
#             offset = pos + offset
#             self._cache_file.seek(offset)

#             byte_size = int().from_bytes(
#                 self._cache_file.read(self.OFFSET_LENGTH),
#                 byteorder='big', signed=False
#             )
#             record_pos_list_bytes = self._cache_file.read(byte_size)
#             sub_record_pos_list = pickle.loads(record_pos_list_bytes)
#             assert isinstance(sub_record_pos_list, list)
#             sub_record_pos_list = [[item[0]+pos, item[1]] for item in sub_record_pos_list]
#             record_pos_list.extend(sub_record_pos_list)
#             pos = self._cache_file.tell()
        
#         self._record_pos_list = record_pos_list

#     def get_record(self, idx):
#         self._check_reopen()
#         record_bytes = self.get_record_bytes(idx)
#         record = pickle.loads(record_bytes)
#         return record

#     def get_record_bytes(self, idx):
#         offset, length = self._record_pos_list[idx]
#         self._sync_lock.acquire()
#         try:
#             self._cache_file.seek(offset)
#             record_bytes = self._cache_file.read(length)
#         finally:
#             self._sync_lock.release()
#         return record_bytes

#     def __len__(self):
#         return len(self._record_pos_list)



# class LRCRecordLoader:
#     def __init__(self, lrc_path):
#         self.loader = ListRecordLoader(lrc_path)

#     def __len__(self):
#         return len(self.loader)
    
#     def get_info(self, idx):
#         table = self.loader.get_record(idx)
#         w = table['image_w']
#         h = table['image_h']
#         n_rows, n_cols = table['layout'].shape
#         n_cells = n_rows * n_cols
#         return w, h, n_cells

#     def get_data(self, idx):
#         table = self.loader.get_record(idx)
#         image = cv2.imread(table['image_path'])
#         return image, table['image_path']

# def get_interval_number(height_list, bins):
#     seg = pd.cut(height_list, bins)
#     counts = pd.value_counts(seg, sort=True)
#     return counts[0]

# def norm_by_height(img, img_save_path=None): 
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     gray_img = np.array(gray_img, dtype=np.uint8)
#     shape = gray_img.shape
#     max_shape = max(shape[0], shape[1])
#     if max_shape > 3000:
#         size = 6
#     elif max_shape > 2000:
#         size = 4
#     elif max_shape > 1000:
#         size = 2
#     else:
#         size = 1
#     height, width = shape[0] // size, shape[1] // size
#     gray_img = cv2.resize(gray_img, (width, height))
#     shape = gray_img.shape
#     width_div_height = shape[1] / shape[0]
#     if width_div_height > 1.5:
#         width_ratio, height_ratio = 20, 10
#     elif width_div_height < 0.67:
#         width_ratio, height_ratio = 10, 20
#     else:
#         width_ratio, height_ratio = 10, 10
#     ratio0, ratio1 = shape[0] // height_ratio, shape[1] // width_ratio

#     is_break = False
#     threshold_num = 120
#     while True:
#         ret, bin_img = cv2.threshold(gray_img, threshold_num, 255, cv2.THRESH_BINARY)
#         bin_img = 255 - bin_img
#         contours, hierarchy = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  
#         contours_array = np.array(contours)
#         if contours_array.size > 5:
#             map_function = lambda x: cv2.boundingRect(x)
#             rect_array = np.array(list(map(map_function, contours_array)))
#             contours_index = np.where((rect_array[:, 2] > 5) & (rect_array[:, 3] > 5) & (rect_array[:, 2] < ratio1) & (rect_array[:, 3] < ratio0))
#             rects = rect_array[contours_index]

#             height_list = rects[:, 3]
#             if height_list.size > 5:
#                 value_count = pd.value_counts(height_list)
#                 if value_count.size < 3:
#                     index_value = value_count.index
#                 else:
#                     index_value = value_count.index[:3]
#                 bins_list = [(i-2, i+1) for i in index_value]
#                 interval_count_list = [get_interval_number(height_list, bins) for bins in bins_list]
#                 result = index_value[np.argmax(interval_count_list)] * size
                
#                 # show the image
#                 if img_save_path:
#                     img = cv2.resize(img, ((width, height)))
#                     for rect in rects:
#                         x1, y1, w, h = rect
#                         cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (255, 0, 255), 1) 
#                     cv2.imwrite(img_save_path, img)

#                 return result

#         if is_break:
#             return 10 * size      # if it can't find the height, return the default value

#         threshold_num += 20
#         if threshold_num > 180:
#             is_break = True


# load_path = '/work1/cv5/hhzhu2/2022/TSR/dataset/lrc_file/test_wireless_table_update20220725_compare.lrc'
# loader = LRCRecordLoader(load_path)


# with open(load_path+'_info.txt', 'w') as f_w:
#     # print(load_path+'_info.txt')
#     for idx in tqdm.tqdm(range(len(loader))):
#         image, image_path = loader.get_data(idx)
#         height = norm_by_height(image)
#         # height_ratio = 16 / height 
#         # height_ratio = 1 if height_ratio <= 0 else height_ratio

#         img_sacle = image.shape[:2]
#         # scale = (int(img_sacle[0]*height_ratio), int(img_sacle[1]*height_ratio))
#         f_w.writelines('{}\t{}\t{}\t{}\n'.format(image_path, img_sacle[0], img_sacle[1], height))
  

# with open(load_path+'_info.txt', 'r') as f:
#     line_list = [line.strip('\n').split() for line in f.readlines()]
#     scale_list = np.array([[line[-3], line[-2]] for line in line_list]).astype(np.int)
#     h_w_max_list = list(map(max, scale_list))
#     num_large = sum([1 if h > 1600 else 0 for h in h_w_max_list])
#     print (num_large, len(line_list), num_large/len(line_list))




