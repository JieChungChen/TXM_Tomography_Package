import os, glob, time
import cv2
import numpy as np
from tqdm import tqdm
from recon_algorithms import mlem_reconstruction_gpu, recon_slice_from_sino
from utils import min_max_normalize


def create_circular_mask(image_size, crop_factor):
    """
    創建圓形遮罩用於重建
    
    在斷層重建中，通常使用圓形遮罩來限制重建區域，
    避免邊緣的偽影影響
    
    Args:
        image_size: 影像大小
        crop_factor: 裁切因子 (0-1)
    
    Returns:
        mask: 圓形遮罩 (image_size, image_size)
    """
    mask = np.zeros((image_size, image_size), dtype=np.float32)
    center = image_size // 2
    radius = image_size // 2 * crop_factor
    
    for i in range(image_size):
        for j in range(image_size):
            # 計算到中心的距離
            r = np.sqrt((i - center)**2 + (j - center)**2)
            if r < radius:
                mask[i, j] = 1.0
    
    return mask


def main():
    input_dir = 'D:/Datasets/TXM_Tomo/Energy1'
    output_dir = 'mle_aligned'
    mle_path = os.path.join(input_dir, output_dir)
    os.makedirs(mle_path, exist_ok=True)
    
    # 影像參數
    image_size = 512         # 影像大小 (512x512)
    
    # 重建參數
    ang_inter = 1            # 角度間隔（度）
    crop_factor = 0.95      # 圓形遮罩的裁切因子
    iter_count = 20         # ML-EM迭代次數
    
    # 重建層範圍（可以選擇只重建部分層）
    start_layer, end_layer = 255, 257
    
    # ==================== 資料準備 ====================
    # 載入所有投影影像
    proj_files = sorted(glob.glob(f'{input_dir}/*.tif'))
    n_proj = len(proj_files)
    img_bulk = np.zeros((n_proj, image_size, image_size), dtype=np.float32)
    
    for i, f in enumerate(tqdm(proj_files, desc="載入投影影像...")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_bulk[i, :, :] = img
    
    # 建立sinogram map
    sinogram_map = img_bulk.transpose((1, 2, 0))
    
    # 創建圓形遮罩
    mask = create_circular_mask(image_size, crop_factor)
    
    # # ==================== ML-EM重建 ====================
    print("\n開始ML-EM重建...")
    start_time = time.time()
    
    # 對指定範圍的每一層進行重建
    for layer in range(start_layer, end_layer):
        print(f"\n處理第 {layer}/{end_layer} 層")
        
        # 提取當前層的sinogram
        sinogram = sinogram_map[layer, :, :]
        print(f"sinogram shape: {sinogram.shape}")
        sinogram = min_max_normalize(sinogram, to_8bit=True)
        sinogram = (255 - sinogram).astype(np.float64)
        
        # 執行ML-EM重建
        reconstruction = mlem_reconstruction_gpu(sinogram, ang_inter, iter_count, mask)
        
        # 也可以使用FBP作為對比
        reconstruction_raw = recon_slice_from_sino(sinogram.T, angle_interval=ang_inter, norm=True)
        
        # 儲存重建結果
        tif_path = os.path.join(mle_path, f'mle_{str(layer).zfill(4)}.tif')
        img_out = min_max_normalize(reconstruction, to_8bit=True)
        cv2.imwrite(tif_path, img_out)

        tif_raw_path = os.path.join(mle_path, f'fbp_{str(layer).zfill(4)}.tif')
        img_out_raw = min_max_normalize(reconstruction_raw, to_8bit=True)
        cv2.imwrite(tif_raw_path, img_out_raw)
        
        print(f"第 {layer} 層完成")
    
    # ==================== 完成 ====================
    elapsed_time = time.time() - start_time
    print(f"\n重建完成!總耗時: {elapsed_time:.2f} 秒")
    print(f"平均每層耗時: {elapsed_time/(end_layer-start_layer+1):.2f} 秒")
    print(f"輸出路徑: {mle_path}")


if __name__ == "__main__":
    main()