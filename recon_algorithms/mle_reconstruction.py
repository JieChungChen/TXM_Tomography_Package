import torch
import torch.nn.functional as F
from tqdm import tqdm


def rotate_torch(image, angle):
    """
    Parameters
    ----------
    image : torch.Tensor
        2D image (image_size, image_size).
    angle : float
        Rotation angle in degrees.
    
    Returns
    -------
    torch.Tensor
        Rotated image with the same shape as input.
    """
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    angle_rad = torch.tensor(angle * torch.pi / 180)

    # rotation matrix
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=torch.float32, device=image.device).unsqueeze(0)  # (1, 2, 3)

    # create affine grid
    N, C, H, W = image.shape
    grid = F.affine_grid(rotation_matrix, size=image.size(), align_corners=False)

    # use grid_sample for interpolation
    rotated_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # remove batch dimension
    return rotated_image.squeeze()


def mlem_reconstruction_gpu(sinogram, ang_inter, iter_count, mask):
    """
    maximum likelihood expectation maximization (MLEM) reconstruction algorithm (GPU accelerated version)
    
    ML-EM is an iterative reconstruction method based on maximum likelihood estimation:
    1. Initialization: Create an initial reconstruction image (usually uniform)
    2. Forward projection: Project the current reconstruction image at various angles to obtain a simulated sinogram
    3. Comparison: Calculate the ratio of the actual sinogram to the simulated sinogram
    4. Backprojection: Backproject the ratio into the image space
    5. Update: Use the backprojection result to update the reconstruction image
    6. Repeat steps 2-5 until convergence or reaching the maximum number of iterations.
    
    Parameters
    ----------
    sinogram : ndarray
        (image_size, n_proj)
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
    n_proj, image_size = sinogram.shape
    sinogram_gpu = torch.from_numpy(sinogram).float().cuda()
    mask_gpu = torch.from_numpy(mask).float().cuda()
    
    # initialize reconstruction image with average value
    mean_value = torch.mean(sinogram_gpu)
    initial_map = torch.ones((image_size, image_size), dtype=torch.float32).cuda() * mean_value / image_size
    
    # initialize simulated sinogram with average value
    simu_sinogram = torch.ones((n_proj, image_size), dtype=torch.float32).cuda() * mean_value
    
    # precompute correction matrix (for handling non-uniform sampling due to rotation)
    correction = torch.ones((image_size, image_size), dtype=torch.float32).cuda()
    R_map_corr = torch.zeros((n_proj, image_size, image_size), dtype=torch.float32).cuda()
    
    for k in range(n_proj):
        angle = k * ang_inter
        R_map_corr_temp = rotate_torch(correction, angle)
        R_map_corr[k, :, :] = R_map_corr_temp
    
    # ML-EM iteration process
    for iteration in tqdm(range(iter_count), desc='ML-EM iteration'):        
        # Step 1: Calculate ratio R = actual sinogram / simulated sinogram
        R = sinogram_gpu / simu_sinogram
        
        # Step 2: Expand ratio to 3D matrix (one layer per angle)
        R_map_temp = torch.repeat_interleave(R[:, None, :], image_size, dim=1)
        
        # Step 3: Backprojection - rotate each angle's ratio back to image space
        R_map = torch.zeros((n_proj, image_size, image_size), dtype=torch.float32).cuda()
        for k in range(n_proj):
            angle = k * ang_inter
            R_map_temp2 = rotate_torch(R_map_temp[k, :, :], angle)
            R_map[k, :, :] = R_map_temp2
        
        # 步驟4: 計算反向投影的總和並進行校正
        # 這是ML-EM的核心更新公式
        R_map_sum = torch.sum(R_map, axis=0) / torch.sum(R_map_corr, axis=0)
        
        # Step 5: Update - multiply the current reconstruction by the backprojected ratio
        initial_map = initial_map * R_map_sum
        
        # Step 6: Forward projection - project the updated image to each angle
        for j in range(n_proj):
            angle = -j * ang_inter
            initial_map_rotate = rotate_torch(initial_map, angle)
            simu_sinogram[j, :] = torch.sum(initial_map_rotate, axis=0)
    
    reconstruction = (initial_map * mask_gpu).cpu().numpy()
    return reconstruction