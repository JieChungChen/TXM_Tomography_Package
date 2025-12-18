import os
import sys
import random
import glob
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import min_max_normalize  # 修改匯入


def add_detector_unevenness(sinogram, block_size_range=(16, 128), variation_range=(0.8, 1.2)):
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
        j = 0
        while j < w:
            # 隨機選擇塊大小
            block_size = np.random.randint(*block_size_range)
            end_j = min(j + block_size, w)
            # 隨機生成增益因子
            gain = np.random.normal(1, 0.05)
            # 應用到整個塊
            uneven_sinogram[i, j:end_j] *= gain
            j = end_j
    
    # 確保值在合理範圍
    uneven_sinogram = np.clip(uneven_sinogram, 0, 1)  # 假設 sinogram 已歸一化到 0-1
    return uneven_sinogram


class Sinogram_Data_Random_Shift(Dataset):
    def __init__(self, data_dir, max_shift, n_projections=181, subfolders=True):
        self.full_proj = n_projections
        self.max_shift = max_shift
        if subfolders:
            sino_folders = glob.glob(f'{data_dir}/*')
            sino_files = [glob.glob(f'{f}/*tif') for f in sino_folders]
            sino_files = sum(sino_files, [])
        else:
            sino_files = sorted(glob.glob(f'{data_dir}/*tif'))
        
        self.sinograms = []
        for f in tqdm(sino_files, desc='load sinograms', dynamic_ncols=True):
            sino_temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
            self.sinograms.append(sino_temp/255)

    def random_shift(self, sino_gt):
        shifted_sino = sino_gt.copy()
        n_proj, size = sino_gt.shape

        # if not 181 degree -> zero padding
        if n_proj < self.full_proj:
            sino_temp = np.zeros((self.full_proj, size))
            sino_temp[:n_proj] = shifted_sino
            shifted_sino = sino_temp.copy()
            sino_gt = sino_temp.copy()

        # add detector unevenness
        shifted_sino = add_detector_unevenness(shifted_sino)

        # add noise
        shifted_sino = np.random.poisson(shifted_sino * 1000) / 1000 

        # apply random shifting
        s_range = random.randint(0, self.max_shift)
        shift = self.generate_shift_curve(181, s_range).astype(int)
        for i in range(self.full_proj):
            shifted_sino[i, :] = np.roll(shifted_sino[i, :], shift[i])
            
        return sino_gt, shifted_sino

    def get_full_sino(self, id, shifted=False):
        sino = self.sinograms[id].copy()
        if shifted:
            return self.random_shift(sino)
        else:
            return sino  
    
    def generate_shift_curve(self, num_proj, max_shift, shift_mode='mixed'):
        x = np.arange(num_proj)
        if shift_mode == 'mixed':
            shift = (
                np.random.uniform(0, max_shift*0.5) * np.sin(2 * np.pi * np.random.uniform(0.02, 0.2) * x) +
                np.random.uniform(-max_shift*0.5, max_shift*0.5, size=num_proj)
            )
        else:
            shift = np.zeros_like(x)
        return shift
    
    def __getitem__(self, index):
        sino = self.sinograms[index].copy()
        return self.random_shift(sino)

    def __len__(self):
        return len(self.sinograms)  

    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = Sinogram_Data_Random_Shift(data_dir='D:/Datasets/Icon_Sino', max_shift=20, n_projections=181, subfolders=True)
    sino_gt, sino_shifted, mask = dataset[5]
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth Sinogram")
    plt.imshow(sino_gt, cmap='gray', vmin=0, vmax=1)
    plt.subplot(1, 2, 2)
    plt.title("Shifted Sinogram")
    plt.imshow(sino_shifted, cmap='gray', vmin=0, vmax=1)
    plt.show()