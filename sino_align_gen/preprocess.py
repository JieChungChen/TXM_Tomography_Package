import random
import glob
import os
import cv2
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm


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
            sino_temp = sino_temp-sino_temp.min(axis=1)[:, None]
            sino_temp = sino_temp/sino_temp.max(axis=1)[:, None]
            sino_temp = 1 - sino_temp
            self.sinograms.append(sino_temp)

    def random_shift(self, sino_gt, apply_mask=True):
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
        return self.random_shift(sino)

    def __len__(self):
        return len(self.sinograms)  
    
    def generate_shift_curve(self, num_proj, max_shift, shift_mode='mixed'):
        x = np.arange(num_proj)
        if shift_mode == 'mixed':
            shift = (
                np.random.normal(0, max_shift*0.5) +
                np.random.uniform(0, max_shift*0.5) * np.sin(2 * np.pi * np.random.uniform(0.05, 0.3) * x) +
                np.random.uniform(-max_shift*0.5, max_shift*0.5, size=num_proj)
            )
        elif shift_mode == 'global':
            shift = np.ones_like(x) * np.random.uniform(-max_shift, max_shift)
        elif shift_mode == 'wave':
            freq = np.random.uniform(0.05, 0.3)
            amp = np.random.uniform(0.3, max_shift)
            shift = amp * np.sin(2 * np.pi * freq * x)
        elif shift_mode == 'jitter':
            shift = np.random.normal(0, 1.5, size=num_proj)
        elif shift_mode == 'block':
            shift = np.zeros_like(x)
            start = np.random.randint(0, num_proj - 30)
            shift[start:start+30] = np.random.uniform(-max_shift, max_shift)
        else:
            shift = np.zeros_like(x)
        return shift