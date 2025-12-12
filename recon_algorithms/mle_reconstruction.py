import torch
import torch.nn.functional as F
import numpy as np


def create_circular_mask(image_size, mask_ratio, device):
    """
    create circular mask for reconstruction, avoiding artifacts at the edges.
    
    Parameters
    ----------
        image_size : int
        crop_factor : float
            ratio of image to keep (0-1)
    
    Returns
    -------
        mask: torch.Tensor
            (image_size, image_size)
    """
    center = image_size // 2
    radius = image_size // 2 * mask_ratio
    
    y, x = torch.meshgrid(torch.arange(image_size, device=device), torch.arange(image_size, device=device), indexing='ij')
    dist = torch.sqrt((x - center)**2 + (y - center)**2)
    mask = (dist < radius).float()
    
    return mask


def rotate_torch_batch(image, angles):
    """
    Parameters
    ----------
    image : torch.Tensor
        2D image (H, W) or 3D (B, H, W).
    angles : list or torch.Tensor
        Rotation angles (degrees).
    
    Returns
    -------
    torch.Tensor
        Batch of rotated images (len(angles)s, H, W).
    """
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image.ndim == 3:
        image = image.unsqueeze(1)  # (B, 1, H, W)
    
    angles_rad = angles * torch.pi / 180
    cos_a = torch.cos(angles_rad)
    sin_a = torch.sin(angles_rad)
    
    # Batch rotation matrices (len(angles), 2, 3)
    rotation_matrices = torch.stack([
        torch.stack([cos_a, -sin_a, torch.zeros_like(cos_a)], dim=1),
        torch.stack([sin_a, cos_a, torch.zeros_like(sin_a)], dim=1)
    ], dim=2).permute(0, 2, 1)  # (len(angles), 2, 3)
    
    N, C, H, W = image.shape
    batch_size = len(angles)
    grid = F.affine_grid(rotation_matrices, size=(batch_size, C, H, W), align_corners=False)
    
    if N == 1:
        image = image.expand(batch_size, -1, -1, -1)
    rotated_images = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
    
    return rotated_images.squeeze()  # (batch_size, H, W)


def mlem_torch_core(sinogram: torch.Tensor, ang_inter: float, iter_count: int, mask: torch.Tensor):
    """
    maximum likelihood expectation maximization (MLEM) reconstruction algorithm (torch accelerated version)
    
    ML-EM is an iterative reconstruction method based on maximum likelihood estimation:
    1. Initialization: Create an initial reconstruction image (usually uniform)
    2. Forward projection: Project the current reconstruction image at various angles to obtain a simulated sinogram
    3. Comparison: Calculate the ratio of the actual sinogram to the simulated sinogram
    4. Backprojection: Backproject the ratio into the image space
    5. Update: Use the backprojection result to update the reconstruction image
    6. Repeat steps 2-5 until convergence or reaching the maximum number of iterations.
    
    Parameters
    ----------
    sinogram : Tensor
        2D image (n_proj, image_size)
    ang_inter : float
        Angle interval in degrees.
    iter_count : int
        Number of iterations.
    mask : Tensor
        Circular mask (image_size, image_size).
    
    Returns
    -------
    reconstruction : Tensor
        (image_size, image_size)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_proj, image_size = sinogram.shape
    sinogram_gpu = sinogram.to(device)
    mask_gpu = mask.to(device)
    
    # initialize reconstruction image with average value
    mean_value = torch.mean(sinogram_gpu)
    initial_map = torch.ones((image_size, image_size), dtype=torch.float32, device=device) * mean_value / image_size
    
    # initialize simulated sinogram with average value
    simu_sinogram = torch.ones((n_proj, image_size), dtype=torch.float32, device=device) * mean_value
    
    # precompute correction matrix (for handling non-uniform sampling due to rotation)
    correction = torch.ones((image_size, image_size), dtype=torch.float32, device=device)
    angles = torch.arange(0, n_proj, device=device) * ang_inter
    R_map_corr = rotate_torch_batch(correction, angles)

    # ML-EM iteration process
    for i in range(iter_count):        
        # Step 1: Calculate ratio R = actual sinogram / simulated sinogram
        R = sinogram_gpu / simu_sinogram
        
        # Step 2: Expand ratio to 3D matrix (one layer per angle)
        R_map_temp = torch.repeat_interleave(R[:, None, :], image_size, dim=1)
        
        # Step 3: Backprojection - rotate each angle's ratio back to image space
        R_map = rotate_torch_batch(R_map_temp, angles)
        
        # step 4: Normalize backprojected result by correction matrix
        # This is the core update formula of ML-EM
        R_map_sum = torch.sum(R_map, dim=0) / torch.sum(R_map_corr, dim=0)
        
        # Step 5: Update - multiply the current reconstruction by the backprojected ratio
        initial_map = initial_map * R_map_sum
        
        # Step 6: Forward projection - project the updated image to each angle
        initial_map_rotate = rotate_torch_batch(initial_map, -angles)
        simu_sinogram = torch.sum(initial_map_rotate, dim=1)
    
    reconstruction = (initial_map * mask_gpu)
    return reconstruction


def mlem_recon(sinogram: np.ndarray, ang_inter: float, iter_count: int, mask_ratio: float):
    """
    maximum likelihood expectation maximization (MLEM) interface function
    
    Parameters
    ----------
    sinogram : ndarray
        (n_proj, image_size)
    ang_inter : int
        Angle interval in degrees.
    iter_count : int
        Number of iterations.
    mask : ndarray
        Circular mask (image_size, image_size).
    
    Returns
    -------
    reconstruction : ndarray
        (image_size, image_size)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    sinogram_torch = torch.tensor(sinogram, device=device).float()
    mask_torch = create_circular_mask(sinogram.shape[1], mask_ratio, device)
    
    reconstruction_torch = mlem_torch_core(sinogram_torch, ang_inter, iter_count, mask_torch)
    reconstruction = reconstruction_torch.cpu().numpy()
    
    return reconstruction