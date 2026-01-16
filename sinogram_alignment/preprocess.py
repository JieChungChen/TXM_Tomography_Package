import random, glob, os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm


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
        self.is_simu = []
        for f in tqdm(sino_files, desc='load sinograms', dynamic_ncols=True):
            if f.endswith('sino.tif'):
                self.is_simu.append(True)
            else:
                if random.random() < 0.65: # randomly skip some real data to balance the dataset
                    continue
                self.is_simu.append(False)

            sino_temp = np.array(Image.open(f)) / 255
            n_proj, size = sino_temp.shape
            if size != 512:
                sino_temp = Image.fromarray(sino_temp).resize((512, n_proj), Image.BILINEAR)
                sino_temp = np.array(sino_temp)
            sino_temp = 1 - sino_temp
            self.sinograms.append(sino_temp)

    def random_shift(self, sino_gt, is_simu, apply_mask=True):
        if is_simu:
            sino_gt = add_detector_unevenness(sino_gt.copy())
            sino_gt = np.random.poisson(sino_gt * 1000) / 1000 
        shifted_sino = sino_gt.copy()
        n_proj, size = sino_gt.shape
        
        # if not 181 degree -> zero padding
        if n_proj < self.full_proj:
            sino_temp = np.zeros((self.full_proj, size))
            sino_temp[:n_proj] = shifted_sino
            shifted_sino = sino_temp.copy()
            sino_gt = sino_temp.copy()

        # apply random shifting
        s_range = random.randint(0, self.max_shift)
        shift = self.generate_shift_curve(181, s_range).astype(int)
        for i in range(self.full_proj):
            shifted_sino[i, :] = np.roll(shifted_sino[i, :], shift[i])

        # random mask partial projections
        mask = np.ones((self.full_proj), dtype=bool)
        if n_proj < self.full_proj:
            mask[:n_proj] = 0
        elif apply_mask:
            end_theta = random.randint(151, 181)
            mask[:end_theta] = 0
            shifted_sino[end_theta:] = 0
            sino_gt[end_theta:] = 0
        else:
            mask = np.zeros((self.full_proj), dtype=bool)
            
        shift = shift/self.max_shift

        return sino_gt, shifted_sino, shift, mask

    def get_full_sino(self, id, shifted=False, apply_mask=False):
        sino = self.sinograms[id].copy()
        if shifted:
            return self.random_shift(sino, apply_mask=apply_mask)
        else:
            return sino  

    def __getitem__(self, index):
        sino = self.sinograms[index].copy()
        is_simu = self.is_simu[index]
        return self.random_shift(sino, is_simu)

    def __len__(self):
        return len(self.sinograms)  
    
    def generate_shift_curve(self, num_proj, max_shift):
        x = np.arange(num_proj)
        shift = (
            np.random.uniform(0, max_shift*0.5) * np.sin(2 * np.pi * np.random.uniform(0.02, 0.2) * x) +
            np.random.uniform(-max_shift*0.5, max_shift*0.5, size=num_proj)
        )
        return shift