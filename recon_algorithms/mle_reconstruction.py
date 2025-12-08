import torch
import torch.nn.functional as F
from tqdm import tqdm


def rotate_torch(image, angle):
    """
    Args:
        image (torch.Tensor): 2D image (H, W).
        angle (float): Rotation angle in degrees.
    
    Returns:
        torch.Tensor: Rotated image with the same shape as input.
    """
    # 確保影像是 4D 張量 (N, C, H, W)
    if image.ndim == 2:
        image = image.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    elif image.ndim == 3:
        image = image.unsqueeze(0)  # (1, C, H, W)

    # 將角度轉換為弧度
    angle_rad = torch.tensor(angle * torch.pi / 180)

    # 計算旋轉矩陣
    cos_a = torch.cos(angle_rad)
    sin_a = torch.sin(angle_rad)
    rotation_matrix = torch.tensor([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0]
    ], dtype=torch.float32, device=image.device).unsqueeze(0)  # (1, 2, 3)

    # 創建 affine grid
    N, C, H, W = image.shape
    grid = F.affine_grid(rotation_matrix, size=image.size(), align_corners=False)

    # 使用 grid_sample 進行插值
    rotated_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='zeros', align_corners=False)

    # 去掉批次維度
    return rotated_image.squeeze(0).squeeze(0)


def mlem_reconstruction_gpu(sinogram, ang_inter, iter_count, mask):
    """
    ML-EM迭代重建算法 (GPU加速版本)
    
    ML-EM是一種基於最大似然估計的迭代重建方法：
    1. 初始化：創建初始重建影像（通常為均勻值）
    2. 前向投影：將當前重建影像投影到各個角度，得到模擬sinogram
    3. 比較：計算實際sinogram與模擬sinogram的比值
    4. 反向投影：將比值反向投影回影像空間
    5. 更新：用反向投影結果更新重建影像
    6. 重複步驟2-5直到收斂
    
    Args:
        sinogram: 輸入sinogram (image_size, image_n)
        ang_inter: 角度間隔（度）
        iter_count: 迭代次數
        mask: 圓形遮罩
    
    Returns:
        reconstruction: 重建結果 (image_size, image_size)
    """
    # 將資料轉移到GPU
    image_size, n_proj= sinogram.shape
    sinogram_gpu = torch.from_numpy(sinogram).float().cuda()
    mask_gpu = torch.from_numpy(mask).float().cuda()
    
    # 初始化重建影像為sinogram的平均值除以影像大小
    mean_value = torch.mean(sinogram_gpu)
    initial_map = torch.ones((image_size, image_size), dtype=torch.float32).cuda() * mean_value / image_size
    
    # 初始化模擬sinogram為平均值
    simu_sinogram = torch.ones((image_size, n_proj), dtype=torch.float32).cuda() * mean_value
    
    # 預計算校正矩陣（用於處理旋轉造成的取樣不均）
    correction = torch.ones((image_size, image_size), dtype=torch.float32).cuda()
    R_map_corr = torch.zeros((image_size, image_size, n_proj), dtype=torch.float32).cuda()
    
    for k in range(n_proj):
        angle = k * ang_inter + 90
        # 旋轉校正矩陣
        R_map_corr_temp = rotate_torch(correction, angle)
        R_map_corr[:, :, k] = R_map_corr_temp
    
    # ML-EM迭代過程
    for iteration in tqdm(range(iter_count), desc='ML-EM iteraion'):        
        # 步驟1: 計算比值 R = 實際sinogram / 模擬sinogram
        R = sinogram_gpu / simu_sinogram
        
        # 步驟2: 將比值擴展到3D矩陣（每個角度一層）
        R_map_temp = torch.repeat_interleave(R[:, None, :], image_size, dim=1)
        
        # 步驟3: 反向投影 - 將每個角度的比值旋轉回影像空間
        R_map = torch.zeros((image_size, image_size, n_proj), dtype=torch.float32).cuda()
        for k in range(n_proj):
            angle = k * ang_inter + 90
            R_map_temp2 = rotate_torch(R_map_temp[:, :, k], angle)
            R_map[:, :, k] = R_map_temp2
        
        # 步驟4: 計算反向投影的總和並進行校正
        # 這是ML-EM的核心更新公式
        R_map_sum = torch.sum(R_map, axis=2) / torch.sum(R_map_corr, axis=2)
        
        # 步驟5: 更新重建影像
        # new_map = initial_map * R_map_sum 是ML-EM的乘法更新規則
        initial_map = initial_map * R_map_sum
        
        # 步驟6: 前向投影 - 將更新後的影像投影到各個角度
        for j in range(n_proj):
            angle = -j * ang_inter - 90
            # 旋轉重建影像到對應角度
            initial_map_rotate = rotate_torch(initial_map, angle)
            # 沿列方向求和得到投影
            simu_sinogram[:, j] = torch.sum(initial_map_rotate, axis=1)
    
    # 返回最終重建結果（轉回CPU）
    reconstruction = (initial_map * mask_gpu).cpu().numpy()
    return reconstruction