import time
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal
from recon_algorithms import mlem_recon
from utils import min_max_normalize


class MLEMWorker(QThread):
    progress = pyqtSignal(int, str)  # 發送進度百分比和剩餘時間
    finished = pyqtSignal(np.ndarray)  # 發送完成的重建結果

    def __init__(self, images, iter_count, mask_ratio, start_layer, end_layer, 
                 angle_interval=1.0, inverse=True):
        """
        ML-EM reconstruction worker thread.

        Parameters
        ----------
        images: np.ndarray
            input sinogram (N, H, W).
        iter_count: int
            number of iterations for ML-EM.
        mask_ratio: float
            mask ratio (0-1).
        start_layer: int
            starting layer for reconstruction.
        end_layer: int
            ending layer for reconstruction.
        angle_interval: float, optional
            projection angle interval (default is 1.0).
        """
        super().__init__()
        self.images = images
        self.iter_count = iter_count
        self.mask_ratio = mask_ratio
        self.start_layer = start_layer
        self.end_layer = end_layer
        self.angle_interval = angle_interval
        self.inverse = inverse
        self.is_cancelled = False

    def cancel(self):
        """Cancel the execution."""
        self.is_cancelled = True

    def run(self):
        """Execute the ML-EM reconstruction logic."""
        n, h, w = self.images.shape
        recon = np.zeros((self.end_layer - self.start_layer, w, w), dtype=np.float32)
        start_time = time.time()

        for i, layer in enumerate(range(self.start_layer, self.end_layer)):
            if self.is_cancelled:
                return

            # Extract the sinogram for the current layer
            sinogram = self.images[:, layer, :]
            sinogram = min_max_normalize(sinogram, to_8bit=True)
            sinogram = (255 - sinogram).astype(np.float64)

            # Execute ML-EM reconstruction
            reconstruction = mlem_recon(sinogram, self.angle_interval, self.iter_count, self.mask_ratio)
            recon[i] = reconstruction

            # Calculate progress and remaining time
            progress = int((i + 1) / (self.end_layer - self.start_layer) * 100)
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
            # Normalize the reconstruction result
            recon -= recon.min()
            recon /= recon.max()
            if self.inverse:
                recon = 1.0 - recon
            recon = (recon * 255).astype(np.uint8)
            self.finished.emit(recon)
