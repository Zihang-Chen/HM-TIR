import argparse
import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
from albumentations import GaussNoise, Lambda, GaussianBlur, RandomBrightnessContrast, Compose

parser = argparse.ArgumentParser(description="Create corrupt images")
parser.add_argument("--dataset_name", default='HM-TIR', type=str,
                    help="dataset_name")
parser.add_argument("--dataset_dir", default=r'./datasets/', type=str, help="train_dataset_dir")
parser.add_argument("--save_dir", default=r"./datasets/corrupt", type=str, help="Save path of checkpoints")

global opt
opt = parser.parse_args()

def stripe_noise(image, **params):
    if random.random() < 0.5:
        g = np.random.randn(1, image.shape[1]) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(1, image.shape[1]) * (np.random.rand() * 55)
    else:
        g = np.random.randn(image.shape[0], 1) * (np.random.rand() * 0.07 + 0.03)
        b = np.random.randn(image.shape[0], 1) * (np.random.rand() * 5)
    if len(image.shape) == 3:
        g = np.expand_dims(g, -1)
        b = np.expand_dims(b, -1)
    noise = image * g + b
    image = np.clip(image.astype("float32") + noise.astype("float32"), 0, 255).astype("uint8")
    return image

def nonuniformity_optical(image, **params):
    h, w = image.shape
    noise = np.ones((h, w)).astype("float32")
    idx_h = np.expand_dims(np.arange(1, h + 1), 1)
    idx_w = np.expand_dims(np.arange(1, w + 1), 0)
    delta = np.random.randint(15, 75 + 1)
    ch = np.random.randint(h)
    cw = np.random.randint(w)

    p = (np.abs(idx_h - ch) ** 2 + np.abs(idx_w - cw) ** 2) ** 0.5
    p /= np.max(p)
    noise *= p
    noise = np.cos(noise * np.pi / 2) ** 4
    if len(image.shape) == 3:
        noise = np.expand_dims(noise, -1)
    if random.random() < 0.5:
        image = np.clip(image.astype("float32") + noise.astype("float32") * delta, 0, 255).astype("uint8")
    else:
        image = np.clip(image.astype("float32") + (1 - noise.astype("float32")) * delta, 0, 255).astype("uint8")
    return image

def Noise(p=1):
    return Compose([
        Lambda(image=nonuniformity_optical, p=1),
        Lambda(image=stripe_noise, p=1),
        GaussNoise(var_limit=(5.0 ** 2,  20.0 ** 2), p=1),
    ], p=p)


def LC(p=1):
    return RandomBrightnessContrast(brightness_limit=(0.2, 0.4), contrast_limit=(-0.8, -0.2), p=p)


def Blur(p=1):
    return Compose([
        GaussianBlur(blur_limit=(7, 23), sigma_limit=(1, 3), p=1)
    ], p=p)


def create_corrupt():
    dataset_dir = opt.dataset_dir + '/' + opt.dataset_name
    with open(dataset_dir + '/test.txt', 'r') as f:
        list = f.read().splitlines()

    tbar = tqdm(list)
    for img_name in tbar:
        img = Image.open((opt.dataset_dir + img_name).replace('//', '/')).convert('L')
        img_list = {'image': np.array(img, dtype=np.uint8)}
        img_list = LC()(**img_list)
        img_list = Blur()(**img_list)
        img_list = Noise()(**img_list)
        cor_img = Image.fromarray(img_list['image']).convert('L')

        if not os.path.exists(opt.save_dir + '/' + opt.dataset_name + '/'):
            os.makedirs(opt.save_dir + '/' + opt.dataset_name + '/')
        cor_img.save(opt.save_dir + '/' + opt.dataset_name + '/' + img_name)


if __name__ == '__main__':
    print("Creating " + opt.dataset_name + " corrupted images ...")
    create_corrupt()
