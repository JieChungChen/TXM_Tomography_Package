import os, glob, random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
from tqdm import tqdm
from recon_algorithms import radon_transform_astra, recon_fbp_astra


def icon_augmentation(icon):
    scale = random.uniform(0.3, 1.0) 
    h, w = icon.shape[:2]
    resized_icon = np.zeros_like(icon)
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


def main():
    icon_root = 'D:/Datasets/Icon_Dataset'
    icon_folders = sorted(glob.glob(os.path.join(icon_root, '*')))

    for folder in icon_folders:
        files = sorted(glob.glob(os.path.join(folder, '*.png')))
        save_folder = folder.replace('Icon_Dataset', 'Icon_Sino')
        os.makedirs(save_folder, exist_ok=True)
        for f in tqdm(files, desc=f"Processing {os.path.basename(folder)}"):
            img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            print(img.shape)
            img = icon_augmentation(img)
            sino = radon_transform_astra(img, num_projections=181)
            sino = ((sino / np.max(sino)) * 210 + 45).astype(np.uint8)
            sino = add_projection_contrast(sino, intensity_range=(0.7, 1.3))
            base_name = os.path.basename(f).replace('.png', '_sino.png')
            cv2.imwrite(os.path.join(save_folder, base_name), sino)
            recon = recon_fbp_astra(sino)
            plt.imshow(recon, cmap='gray')
            plt.title('Reconstructed Image')
            plt.show()
            plt.close()



if __name__ == '__main__':
    main()