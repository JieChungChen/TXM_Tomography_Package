import numpy as np
from src.logic.utils import mosaic_stitching, norm_to_8bit, image_resize


class TXM_Images:
    def __init__(self, images: np.ndarray, mode, metadata=None, angles=None):
        """
        TXM image container.

        Parameters
        ----------
        images : np.ndarray
            3D NumPy array of shape (N, H, W)
        mode : str
            'tomo' or 'mosaic'
        metadata : dict, optional
            Metadata dictionary
        angles : np.ndarray, optional
            Array of rotation angles for tomography images
        """
        self.original = images.copy()  # Keep a backup for restoring raw images
        self.images = images  # shape: (N, H, W)
        self.mode = mode
        self.metadata = metadata or {}
        self.ref = None
        self.shift_array = None

        if mode == 'tomo':
            if angles is None:
                self.angles = np.linspace(0, len(images) - 1, len(images))
            else:
                self.angles = angles
        
    def __len__(self):
        return self.images.shape[0]

    def get_image(self, idx):
        return self.images[idx].copy()

    def get_theta(self, idx):
        if self.mode == 'tomo':
            return self.angles[idx]
        return None
    
    def get_full_images(self):
        return self.images
    
    def get_norm_images(self):
        norm_images = np.zeros_like(self.images, dtype=np.uint8)
        for i in range(self.images.shape[0]):
            norm_images[i] = norm_to_8bit(self.images[i])
        return norm_images
    
    def get_mosaic(self):
        if self.mode == 'mosaic':
            rows, cols = self.metadata['mosaic_row'], self.metadata['mosaic_column']
            mosaic = mosaic_stitching(self.images, rows, cols)
            mosaic = norm_to_8bit(mosaic, clip_lower=0., clip_upper=0.)
            return mosaic
        return None
    
    def set(self, idx, image):
        """
        set image at specified index.
        """
        self.images[idx] = image

    def set_full_images(self, images):
        """
        set the entire images array.
        """
        self.images = images

    def set_shift_array(self, shift_array):
        """
        set the shift array for all images.

        Parameters
        ----------
        shift_array : np.ndarray
            Array of shape (N, 2) where each row is (y_shift, x_shift)
        """
        self.shift_arrays = shift_array

    def flip_vertical(self):
        self.images = np.flip(self.images, axis=1)

    def apply_y_shift(self, shift_value):
        """
        apply vertical shift to all images.

        Parameters
        ----------
        shift_value : int
            The amount of vertical shift. Positive values shift down, negative values shift up.
        """
        self.images = np.roll(self.images, shift_value, axis=1)

    def apply_ref(self, ref_image1, ref_image2=None, split_point=None):
        """
        apply reference image(s) to the TXM images.

        Parameters
        ----------
        ref_image1 : np.ndarray
            The first reference image.
        ref_image2 : np.ndarray, optional
            The second reference image.
        split_point : int, optional
            The index to split the images for dual reference application.
        """
        if ref_image1 is not None and ref_image2 is None and split_point is None:
            # Single reference
            self.ref = ref_image1
            ref_resized = image_resize(ref_image1, self.images.shape[-1])
            self.images = self.original / ref_resized
        elif ref_image1 is not None and ref_image2 is not None and split_point is not None:
            # Dual reference
            imgs = np.zeros_like(self.original, dtype=np.float32)
            # First half
            ref1_resized = image_resize(ref_image1, imgs.shape[-1])
            imgs[:split_point] = self.original[:split_point] / ref1_resized
            # Second half
            ref2_resized = image_resize(ref_image2, imgs.shape[-1])
            imgs[split_point:] = self.original[split_point:] / ref2_resized
            self.images = imgs
        else:
            pass
