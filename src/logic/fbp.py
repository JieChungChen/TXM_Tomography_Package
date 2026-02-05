import time
import numpy as np
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal


class FBPWorker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(np.ndarray)
    def __init__(self, images, angles, target_size, angle_interval=1.0, astra_available=False, inverse=True):
        """
        FBP worker thread for reconstruction.
        
        Parameters
        ----------
        images: projection images (N, H, W)
        angles: rotation angles for each projection (can be None)
        target_size: target reconstruction resolution (int)
        angle_interval: angle interval (degrees, default 1.0)
        astra_available: whether ASTRA GPU acceleration is available (bool)
        inverse: whether to perform inverse FBP (bool, default True)
        """
        super().__init__()
        self.is_cancelled = False
        self.inverse = inverse
        self.angle_interval = angle_interval
        self.astra_available = astra_available

        if self.astra_available:
            try:
                from recon_algorithms import recon_fbp_astra
                self.recon_fbp_astra = recon_fbp_astra
            except ImportError:
                self.astra_available = False

        # Create angle sequence based on the number of images and angle interval.
        n_images = len(images)
        if angles is None or len(angles) == 0:
            # Create angles based on interval, assuming scan range from -90 to +90.
            self.angles = np.arange(n_images) * angle_interval - 90.0
        else:
            # Use provided angles, but reconstruct angle sequence based on number of images and interval.
            self.angles = np.arange(n_images) * angle_interval + angles[0]

        # If target resolution is specified, resize images.
        self.images = []
        for i in range(len(images)):
            img_temp = Image.fromarray(images[i])
            img_temp = img_temp.resize((target_size, target_size), Image.Resampling.LANCZOS)
            self.images.append(np.array(img_temp))
        self.images = np.array(self.images)

    def cancel(self):
        self.is_cancelled = True

    def run(self):
        n, h, w = self.images.shape
        recon = np.zeros((h, w, w))
        start_time = time.time()

        if self.is_cancelled:
            return
        
        if self.astra_available:
            for i in range(h):
                if self.is_cancelled:
                    break 

                sino = self.images[:, i, :]
                temp = self.recon_fbp_astra(sino, angle_interval=self.angle_interval, norm=False) 
                recon[i] = temp
            
                progress = int((i + 1) / h * 100)
                elapsed = time.time() - start_time
                if progress > 0:
                    total_est = elapsed / (progress / 100.0)
                    remaining = total_est - elapsed
                    mins, secs = divmod(int(remaining), 60)
                    remaining_str = f"Estimated time left: {mins}m {secs}s"
                else:
                    remaining_str = ""
                
                self.progress.emit(progress, remaining_str)

        else: 
            img_size_padded = max(64, 2 ** int(np.ceil(np.log2(2 * w))))
            hann = get_hann_filter(img_size_padded)
            center, x, y, cos_vals, sin_vals = prepare_fbp_geometry(w, self.angles)

            empty_sino = np.ones((n, w))
            recon_0 = filter_back_projection_fast(empty_sino, cos_vals, sin_vals, center, x, y, hann)

            for i in range(h):
                if self.is_cancelled:
                    break 

                sino = self.images[:, i, :]
                temp = filter_back_projection_fast(sino, cos_vals, sin_vals, center, x, y, hann, filtered=True, circle=False)
                
                temp /= recon_0
                recon[i] = temp
                
                progress = int((i + 1) / h * 100)
                elapsed = time.time() - start_time
                if progress > 0:
                    total_est = elapsed / (progress / 100.0)
                    remaining = total_est - elapsed
                    mins, secs = divmod(int(remaining), 60)
                    remaining_str = f"Estimated time left: {mins}m {secs}s"
                else:
                    remaining_str = ""
                
                self.progress.emit(progress, remaining_str)

        if not self.is_cancelled:
            recon -= recon.min()
            recon /= recon.max()
            if self.inverse:
                recon = 1 - recon
            recon = (recon * 255).astype(np.uint8)
            self.finished.emit(recon)


# ---- FBP core functions; don't modify unless you know what you are doing ---- #
def get_hann_filter(img_size):
    """濾波器預計算。"""
    n = np.concatenate((
        np.arange(1, img_size//2 + 1, 2),
        np.arange(img_size//2 - 1, 0, -2)
    ))
    f = np.zeros(img_size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n) ** 2
    ramp = 2 * np.real(np.fft.fft(f))
    hann = np.hanning(img_size)
    hann = np.fft.fftshift(hann)
    return ramp * hann


def prepare_fbp_geometry(L, angles_deg):
    """準備FBP幾何參數。"""
    center = L // 2
    x, y = np.meshgrid(np.arange(L) - center, np.arange(L) - center, indexing='ij')
    angles = np.deg2rad(angles_deg)
    cos_vals = np.cos(angles + np.pi/2)
    sin_vals = np.sin(angles + np.pi/2)
    return center, x, y, cos_vals, sin_vals


def filter_back_projection_fast(sino, cos_vals, sin_vals, center, x, y, hann=None, filtered=True, circle=False):
    """快速反投影演算法實作。"""
    n_proj, L = sino.shape

    if filtered and hann is not None:
        pad_width = hann.size - L
        sino_padded = np.pad(sino, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        sino_fft = np.fft.fft(sino_padded, axis=-1)
        sino = np.real(np.fft.ifft(sino_fft * hann, axis=-1))[:, :L]

    recon = np.zeros((L, L), dtype=np.float32)

    for i in range(n_proj):
        t = x * cos_vals[i] + y * sin_vals[i]
        t_idx = np.round(t + center).astype(np.int32)
        valid = (t_idx >= 0) & (t_idx < L)
        recon[valid] += sino[i, t_idx[valid]]

    if circle:
        Y, X = np.ogrid[:L, :L]
        dist_from_center = np.sqrt((X - center) ** 2 + (Y - center) ** 2)
        mask = dist_from_center > center
        recon[mask] = 0

    return recon