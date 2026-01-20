import numpy as np
from src.logic.utils import mosaic_stitching, norm_to_8bit, image_resize


class TXM_Images:
    """TXM 影像容器，包含中繼資料與處理方法。"""

    def __init__(self, images: np.ndarray, mode, metadata=None, angles=None):
        """
        初始化 TXM 影像容器。

        Args:
            images: 3D NumPy 陣列，形狀為（N, H, W）
            mode: 'tomo' 或 'mosaic'
            metadata: 中繼資料字典
            angles: 斷層影像的旋轉角度陣列
        """
        self.images = images  # shape: (N, H, W)
        self.original = images.copy()  # 保留備份以便還原參考影像
        self.mode = mode
        self.metadata = metadata or {}
        self.ref = None

        if mode == 'tomo':
            if angles is None:
                self.angles = np.linspace(0, len(images) - 1, len(images))
            else:
                self.angles = angles
        
    def __len__(self):
        return self.images.shape[0]

    def get_image(self, idx):
        return self.images[idx]

    def get_theta(self, idx):
        if self.mode == 'tomo':
            return self.angles[idx]
        return None
    
    def get_array(self):
        return self.images
    
    def get_mosaic(self):
        if self.mode == 'mosaic':
            rows, cols = self.metadata['mosaic_row'], self.metadata['mosaic_column']
            mosaic = mosaic_stitching(self.images, rows, cols)
            mosaic = norm_to_8bit(mosaic, clip_percent=0.)
            return mosaic
        return None
    
    def set(self, idx, image):
        self.images[idx] = image

    def flip_vertical(self):
        self.images = np.flip(self.images, axis=1)

    def y_shift(self, shift_amount):
        self.images = np.roll(self.images, shift_amount, axis=1)

    def apply_ref(self, ref_image):
        if ref_image is not None:
            self.ref = ref_image
            if self.images.shape[-1] != self.ref.shape[-1]:
                self.images = self.original / image_resize(ref_image, self.images.shape[-1])
            else:
                self.images = self.original / ref_image

