import os
import glob
import time
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from recon_algorithms import mlem_recon, recon_fbp_astra
from utils import min_max_normalize, from_xml


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


def parse_args():
    parser = argparse.ArgumentParser(description="ML-EM Reconstruction")
    parser.add_argument('--input_dir', type=str, default='D:/Datasets/TXM_Tomo/tomo4-E-b2-60s-181p')
    parser.add_argument('--output_dir', type=str, default='D:/Datasets/MLE_AI')
    parser.add_argument('--save_fbp', type=bool, default=False, help='Whether to save FBP reconstruction results.')
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--ang_inter', type=float, default=1.0, help='Angle interval in degrees.')
    parser.add_argument('--crop_factor', type=float, default=0.95, help='Crop factor for circular mask (0-1).')
    parser.add_argument('--iter_count', type=int, default=100, help='Number of ML-EM iterations.')
    parser.add_argument('--start_layer', type=int, default=210)
    parser.add_argument('--end_layer', type=int, default=455)
    return parser.parse_args()


def main(args):
    os.makedirs(args.output_dir + '/mle', exist_ok=True)
    os.makedirs(args.output_dir + '/fbp', exist_ok=True)
    xml_path = args.input_dir + '/annotations.xml'
    sample_name = os.path.basename(args.input_dir)
    
    # ==================== 資料準備 ====================
    # 載入所有投影影像
    proj_files = sorted(glob.glob(f'{args.input_dir}/*.tif'))
    size = args.image_size
    start_layer, end_layer = args.start_layer, args.end_layer
    tomo = []
    
    for i, f in enumerate(tqdm(proj_files, desc="載入投影影像...")):
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(np.float32)
        tomo.append(img)

    tomo = np.array(tomo) 
    n_proj, height, width = tomo.shape

    if os.path.exists(xml_path):
        center_ps = from_xml(xml_path)
        box_height = center_ps[0][2]
        box_height = (size/height) * box_height * 0.9
        start_layer, end_layer = 256 - int(box_height // 2), 256 + int(box_height // 2)
        for i in range(n_proj):
            x_shift = int(width // 2 - center_ps[i][0])
            y_shift = int(height // 2 - center_ps[i][1])
            proj_temp = np.roll(tomo[i], (y_shift, x_shift), axis=(0, 1))
            tomo[i] = proj_temp

    if height != size or width != size:
        tomo = np.array([cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR) for img in tomo])
    
    # 創建圓形遮罩
    mask = create_circular_mask(size, args.crop_factor)
    
    # # ==================== ML-EM重建 ====================
    print("\n開始ML-EM重建...")
    start_time = time.time()
    
    # 對指定範圍的每一層進行重建
    pbar = tqdm(range(start_layer, end_layer, 3))
    for layer in pbar:
        # 提取當前層的sinogram
        sinogram = tomo[:, layer, :]
        sinogram = min_max_normalize(sinogram, to_8bit=True)
        sinogram = (255 - sinogram).astype(np.float64)
        pbar.set_description(f"Processing layer {layer}/{end_layer}, sino shape {sinogram.shape}")
        
        # 執行ML-EM重建
        reconstruction = mlem_recon(sinogram, args.ang_inter, args.iter_count, mask)
        tif_path = os.path.join(args.output_dir, f'mle/{sample_name}_{str(layer).zfill(4)}.tif')
        img_out = min_max_normalize(reconstruction, to_8bit=True)
        cv2.imwrite(tif_path, img_out)
        
        # 使用FBP重建作為比較
        if args.save_fbp:
            reconstruction_raw = recon_fbp_astra(sinogram, angle_interval=args.ang_inter, norm=True)
            tif_raw_path = os.path.join(args.output_dir, f'fbp/{sample_name}_{str(layer).zfill(4)}.tif')
            img_out_raw = min_max_normalize(reconstruction_raw, to_8bit=True)
            cv2.imwrite(tif_raw_path, img_out_raw)
            
    # ==================== 完成 ====================
    elapsed_time = time.time() - start_time
    print(f"\n重建完成!總耗時: {elapsed_time:.2f} 秒")
    print(f"輸出路徑: {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)