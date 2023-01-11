import albumentations as A
import numpy as np 
import os
import random
import cv2

class HorizontalFlip:

    def __init__(self, p=0.5):
        self.aug = A.HorizontalFlip(p=p)

    def __call__(self, img, lbl):
        augmented = self.aug(image=img, mask=lbl)
        return augmented["image"], augmented["mask"]

class VerticalFlip:

    def __init__(self, p=0.5):
        self.aug = A.VerticalFlip(p=p)

    def __call__(self, img, lbl):
        augmented = self.aug(image=img, mask=lbl)
        return augmented["image"], augmented["mask"]

class RandomCrop:

    def __init__(self, size):
        h = size[0]
        w = size[1]
        self.aug = A.RandomCrop(h, w)

    def __call__(self, img, lbl):
        augmented = self.aug(image=img, mask=lbl)
        return augmented["image"], augmented["mask"]

class RandomScale:

    def __init__(self, scales=[0.5, 0.75, 1, 1.5, 2]):
        self.scales = scales

    def __call__(self, img, lbl):
        scale = random.choice(self.scales)
        new_h = int(img.shape[0] * scale)
        new_w = int(img.shape[1] * scale)
        transform = A.Resize(new_h, new_w)
        augmented = transform(image=img, mask=lbl)
        return augmented["image"], augmented["mask"]

def convert_to_color(image):
    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]
    new = np.ones((image.shape[0], image.shape[1], 3)) * 255
    for i in range (19):
        new[image == i, 0] = colors[i][0]
        new[image == i, 1] = colors[i][1]
        new[image == i, 2] = colors[i][2]
    return new.astype(np.uint8)

if __name__ == "__main__":
    from PIL import Image
    img = np.asarray(Image.open("D:\\Computer vision\\DA&Segmen\\UDACode\\GTAV\\images\\00001.png"))
    lbl = np.asarray(Image.open("D:\\Computer vision\\DA&Segmen\\UDACode\\GTAV\\labels\\00001.png"))
    h, w = img.shape[0], img.shape[1]
    augment = RandomScale([0.5, 0.75, 1])
    new_img, new_lbl = augment(img, lbl)
    img_save = np.ones((h, w*2+100, 3)) * 255
    lbl_save = np.ones((h, w*2+100)) * 255
    img_save[:h, :w, :] = img 
    img_save[:new_img.shape[0], w+100:w+100+new_img.shape[1],:] = new_img
    lbl_save[:h, :w] = lbl
    lbl_save[:new_lbl.shape[0], w+100:w + 100 + new_lbl.shape[1]] = new_lbl
    image = Image.fromarray(img_save.astype(np.uint8))
    lbl_save = convert_to_color(lbl_save)
    label = Image.fromarray(lbl_save)
    image.save("D:\\Computer vision\\DA&Segmen\\UDACode\\visualization\\image.png")
    label.save("D:\\Computer vision\\DA&Segmen\\UDACode\\visualization\\label.png")
