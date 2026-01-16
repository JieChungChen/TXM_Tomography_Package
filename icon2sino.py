import os
import glob
import argparse
import random
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from recon_algorithms import recon_fbp_astra, radon_transform_torch
from utils import min_max_normalize


def icon_augmentation(icon):
    scale = random.uniform(0.1, 0.5) 
    h, w = icon.shape[:2]
    resized_icon = np.zeros_like(icon) + icon[0, 0]
    new_w, new_h = int(w * scale), int(h * scale)
    icon = cv2.resize(icon, (new_w, new_h), interpolation=cv2.INTER_AREA)
    resized_icon[(h - new_h) // 2:(h - new_h) // 2 + new_h, (w - new_w) // 2:(w - new_w) // 2 + new_w] = icon
    return resized_icon


def add_projection_contrast(image, intensity_range=(0.5, 1.5)):
    """
    對 sinogram 的部分投影進行隨機對比度變化。
    :param image: 輸入的 sinogram 圖像 (numpy array)
    :param num_projections: 需要改變對比度的投影數量
    :param intensity_range: 對比度變化的範圍 (0.5 表示減弱，1.5 表示增強)
    :return: 對比度變化後的圖像
    """
    contrasted_image = image.copy()
    h, w = image.shape[:2]
    # 隨機選擇一列 (投影)
    start_id, end_id = np.random.randint(0, h-1, 2)
    # 隨機選擇對比度增強或減弱的強度
    intensity = random.uniform(*intensity_range)
    # 對該列進行對比度變化
    contrasted_image[start_id:end_id, :] = np.clip(
        contrasted_image[start_id:end_id, :] * intensity, 0, 255
    ).astype(np.uint8)
    return contrasted_image


def add_background_texture(image):
    """
    添加隨機背景紋理。
    :param image: 輸入的圖像 (numpy array)
    :return: 添加背景紋理後的圖像
    """
    h, w = image.shape[:2]
    noise = np.random.normal(0, 10, (h, w)).astype(np.float32)
    textured_image = image.astype(np.float32) + noise
    textured_image = np.clip(textured_image, 0, 255).astype(np.uint8)
    return textured_image


def add_detector_unevenness(sinogram, block_size_range=(16, 48), variation_range=(0.9, 1.1)):
    """
    模擬檢測器像素感應不均的效果，對每個 projection 的像素分成隨機大小的塊，每個塊應用隨機增益。
    
    :param sinogram: 輸入的 sinogram (numpy array, shape: (n_angles, image_size))
    :param block_size_range: 塊大小範圍 (e.g., (8, 32) 表示 8-32 像素寬)
    :param variation_range: 每個塊的隨機增益範圍 (e.g., 0.8-1.2 表示 ±20% 變化)
    :return: 添加不均勻效果後的 sinogram
    """
    h, w = sinogram.shape  # (n_angles, image_size)
    uneven_sinogram = sinogram.copy().astype(np.float32)
    
    # 對每個 projection (每一行)，將寬度分成隨機大小的塊
    for i in range(h):  # 遍歷每個角度 (projection)
        # 隨機控制點位置
        step = np.random.randint(*block_size_range)
        ctrl_x = np.arange(0, w + step, step)
        if ctrl_x[-1] < w:  # 確保包含尾端
            ctrl_x = np.append(ctrl_x, w)
        # 隨機控制點增益
        ctrl_y = np.random.uniform(variation_range[0], variation_range[1], size=ctrl_x.shape[0])
        # 線性插值為平滑曲線
        gain_curve = np.interp(np.arange(w), ctrl_x, ctrl_y)
        # 可選：再用高斯平滑
        uneven_sinogram[i] *= gain_curve
    
    # 確保值在合理範圍
    uneven_sinogram = np.clip(uneven_sinogram, 0, 1)  # 假設 sinogram 已歸一化到 0-1
    return uneven_sinogram


def parse_args():
    parser = argparse.ArgumentParser(description="Icon to TXM Sinogram Generator")
    parser.add_argument('--input_dir', type=str, default='D:/Datasets/Icon_Dataset')
    parser.add_argument('--output_dir', type=str, default='D:/Datasets/Icon_Sino')
    parser.add_argument('--image_size', type=int, default=732)
    return parser.parse_args()


def main(args):
    icon_folders = sorted(glob.glob(os.path.join(args.input_dir, '*')))
    target_size = args.image_size

    for folder in icon_folders:
        exts = ('*.png', '*.jpg', '*.jpeg')
        files = sorted([p for ext in exts for p in glob.glob(os.path.join(folder, ext))])
        save_path = os.path.join(args.output_dir, os.path.basename(folder))
        os.makedirs(save_path, exist_ok=True)
        for f in tqdm(files, desc=f"Processing {os.path.basename(folder)}"):
            img = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            alpha_channel = img[..., -1]
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            background = alpha_channel == 0
            gray_avg = np.mean(img[~background]).astype(int)
            img[background] = random.randint(gray_avg-10, gray_avg+10)
            if img.shape[0] !=  target_size or img.shape[1] != target_size:
                img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

            # generate txm-like sinogram
            img = icon_augmentation(img)
            sino = radon_transform_torch(img, np.linspace(0, np.pi, 181), circle=False, sub_sample=0.7)
            sino = min_max_normalize(sino)
            
            sino = add_detector_unevenness(sino)

            sino = np.random.poisson(sino * 1000) / 1000 
            # sino = add_projection_contrast(sino, intensity_range=(0.7, 1.3))

            # save sinogram
            base_name, ext = os.path.splitext(os.path.basename(f))
            new_filename = base_name + '_sino.tif'
            sino = np.clip(sino, 0, 1)
            sino = 1 - sino
            sino = (sino * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(save_path, new_filename), sino)
            
            # visualize
            recon = recon_fbp_astra(sino, angle_interval=1, norm=True)
            plt.imshow(recon, cmap='gray')
            plt.title('Reconstructed Image')
            plt.show()
            plt.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)