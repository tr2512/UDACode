import os
import torch
import numpy as np
from PIL import Image
from glob import glob

class GTA5(torch.utils.data.Dataset):

    def __init__(self, opt, logger, augmentations=None, randaug=None, ignore_index=255):
        self.opt = opt
        self.logger = logger
        self.agumentations = augmentations
        self.randaug = randaug
        self.ignore_index = ignore_index
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        folds = glob(opt.src_root + "/*" )
        img_list = []
        for fold in folds:
            l = glob(fold + "/images/*.png")
            img_list += l
        
    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_dir = self.img_list[idx]
        lbl_dir = img_dir.replace("images", "labels")
        img = np.array(Image.open(os.path.join(img_dir)), dtype=np.float32)
        lbl = np.array(Image.open(os.path.join(lbl_dir), dtype=np.int_))
        lbl = self.encode_segmap(lbl)
        data = {}
        if self.augmentations:
            img, lbl = self.augmentations(img, lbl)
        data['img'] = img
        data['lbl'] = lbl
        data['dir'] = img_dir
        if self.randaug:
            imgS, lblS = self.randaug(img, lbl)
            data["imgS"] = imgS
            data["lblS"] = lblS
        return data
        
    def encode_segmap(self, lbl):
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl