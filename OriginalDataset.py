# -*- coding: utf-8 -*-
import torch
import random
import numpy as np
import os
import openslide


# map vips formats to np dtypes
format_to_dtype = {
    'uchar': np.uint8,
    'char': np.int8,
    'ushort': np.uint16,
    'short': np.int16,
    'uint': np.uint32,
    'int': np.int32,
    'float': np.float32,
    'double': np.float64,
    'complex': np.complex64,
    'dpcomplex': np.complex128,
}

DATA_PATH = f'data_directry'

class OriginalDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform = None, bag_num=50, bag_size=100, train=True):
        self.transform = transform
        
        self.bag_list = []
        for slide_data in dataset:
            slideID = slide_data[0]
            label = slide_data[1]
            pos = np.loadtxt(f'{DATA_PATH}/csv/{slideID}.csv', delimiter=',', dtype='int')
            if not train:
                np.random.seed(seed=int(slideID.replace('SLIDE_','')))
            np.random.shuffle(pos)
            if pos.shape[0] > bag_num*bag_size:
                pos = pos[0:(bag_num*bag_size),:]
                for i in range(bag_num):
                    patches = pos[i*bag_size:(i+1)*bag_size,:].tolist()
                    self.bag_list.append([patches, slideID, label])
            else:
                for i in range(pos.shape[0]//bag_size):
                    patches = pos[i*bag_size:(i+1)*bag_size,:].tolist()
                    self.bag_list.append([patches, slideID, label])

        random.shuffle(self.bag_list)
        self.data_num = len(self.bag_list)

    def __len__(self):
        return self.data_num
        
    def __getitem__(self, idx):
        pos_list = self.bag_list[idx][0]
        patch_len = len(pos_list)
        b_size = 224
        svs_list = os.listdir(f'{DATA_PATH}/svs')
        svs_fn = [s for s in svs_list if self.bag_list[idx][1] in s]
        svs = openslide.OpenSlide(f'{DATA_PATH}/svs/{svs_fn[0]}')
        bag = torch.empty(patch_len, 3, 224, 224, dtype=torch.float)

        i = 0
        # 画像読み込み
        for pos in pos_list:
            if self.transform:
                img = svs.read_region((pos[0],pos[1]),0,(b_size,b_size)).convert('RGB')
                img = self.transform(img)
                bag[i] = img
            i += 1

        label = self.bag_list[idx][2]
        label = torch.LongTensor([label])

        # バッグ, ラベルを返す
        return bag, label
