import tqdm
import copy
import random
import numpy as np
from libs.utils import logger
from collections import defaultdict


class BucketSampler:
    def __init__(self, dataset, img_size_dict, world_size, rank_id, fix_batch_size=None, max_pixel_nums=None, max_row_nums=None, max_col_nums=None, max_batch_size=8, min_batch_size=1, seps=(100, 100, 5, 5, 10), epoch=0):
        self.dataset = dataset
        self.img_size_dict = img_size_dict
        self.world_size = world_size
        self.rank_id = rank_id
        self.seps = seps
        self.fix_batch_size = fix_batch_size
        self.max_batch_size = max_batch_size
        self.min_batch_size = min_batch_size
        self.max_pixel_nums = max_pixel_nums
        self.max_row_nums = max_row_nums
        self.max_col_nums = max_col_nums
        assert (fix_batch_size is not None) or (max_pixel_nums is not None)
        self.cal_buckets()
        self.seed = 20
        self.epoch = epoch

    def count_keys(self):
        infos = []
        for i in tqdm.tqdm(range(len(self.dataset))):
            info = self.dataset.get_info(i)
            h, w = self.img_size_dict[info[0]]
            infos.append([ w, h, info[-3], info[-2], info[-1]])
        return infos

    def cal_buckets(self):
        infos = self.count_keys()
        min_sizes = tuple(np.array(infos).min(0))
        max_sizes = tuple(np.array(infos).max(0))
        assert len(self.seps) == len(min_sizes)
        print('train dataset (w,h,r,c,cell), max sizes: {}, min size: {}'.format(max_sizes, min_sizes))

        buckets = defaultdict(list)
        for idx, info in enumerate(infos):
            bucket_idxes = list()
            for sep, size, min_size in zip(self.seps, info, min_sizes):
                bucket_idx = (size - min_size) // sep
                bucket_idxes.append(str(bucket_idx))
            bucket_idxes = '-'.join(bucket_idxes)
            buckets[bucket_idxes].append(idx)

        valid_buckets = dict()
        for bucket_key, bucket_samples in buckets.items():
            if len(bucket_samples) < self.min_batch_size:
                continue
            if (self.fix_batch_size is not None) and (len(bucket_samples) < self.fix_batch_size):
                continue

            w, h, n_rows, n_cols, n_cells = [(int(item) + 1) * sep + min_size for item, min_size, sep in zip(bucket_key.split('-'), min_sizes, self.seps)]
            if self.fix_batch_size is not None:
                if h * w * self.fix_batch_size > self.max_pixel_nums:
                    continue
            else:
                if h * w * self.min_batch_size > self.max_pixel_nums: # remove too large pixel
                    continue

            if n_rows > 40:#self.max_row_nums: # remove too many rows
                continue
            if n_cols > 40:#self.max_col_nums: # remove too many cols
                continue
            
            if self.fix_batch_size is not None:
                batch_size = self.fix_batch_size
            else:
                image_batch_size = min(self.max_batch_size, max(self.max_pixel_nums // (w*h), self.min_batch_size), len(bucket_samples))
                row_batch_size = min(self.max_batch_size, max(self.max_row_nums // n_rows, self.min_batch_size), len(bucket_samples))
                col_batch_size = min(self.max_batch_size, max(self.max_col_nums // n_cols, self.min_batch_size), len(bucket_samples))
                batch_size = min(image_batch_size, row_batch_size, col_batch_size)

            valid_buckets[bucket_key] = dict(
                samples=bucket_samples,
                batch_size=batch_size
            )

        logger.info('Total %d buckets' % (len(valid_buckets)))
        self.buckets = [valid_buckets[bucket_key] for bucket_key in sorted(valid_buckets.keys())]
        total_nums = len(infos)
        valid_nums = sum([len(item['samples']) for item in valid_buckets.values()])
        logger.info('Total %d samples, but ignore %d samples' % (total_nums, total_nums - valid_nums))

    def __iter__(self):
        random_inst = random.Random(self.seed + self.epoch)
        batches = list()
        for bucket in self.buckets:
            sample = copy.deepcopy(bucket['samples'])
            batch_size = bucket['batch_size']
            random_inst.shuffle(sample)
            idx = 0
            while idx < len(sample):
                batch = sample[idx:idx+batch_size]
                idx += batch_size
                if len(batch) < self.min_batch_size:
                    continue
                batches.append(batch)
        random_inst.shuffle(batches)
        
        align_nums = (len(batches) // self.world_size) * self.world_size
        batches = batches[: align_nums]
        for batch_idx in range(self.rank_id, len(batches), self.world_size):
            yield batches[batch_idx]

    def __len__(self):
        batch_nums = 0
        for bucket in self.buckets:
            bucket_sample_nums = len(bucket['samples'])
            bucket_bs = bucket['batch_size']
            batch_nums += bucket_sample_nums//bucket_bs
            if bucket_sample_nums % bucket_bs >= self.min_batch_size:
                batch_nums += 1
        
        return batch_nums//self.world_size

    def set_epoch(self, epoch):
        self.epoch = epoch
