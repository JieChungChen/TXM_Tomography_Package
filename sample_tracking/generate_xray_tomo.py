"""
生成符合 X-ray 物理的合成 tomography 數據
- 複雜幾何形狀（多個球體組合）
- 正確的 X-ray 造影（背景亮，物體暗）
"""

import numpy as np
import cv2
import os
from tqdm import tqdm
import xml.etree.ElementTree as ET
from xml.dom import minidom
from scipy.ndimage import rotate


def create_clustered_3d_volume(size=128, img_size=256, n_clusters=None, n_objects_per_cluster=None):
    """創建多群球體的 3D 體積，bbox 只框其中一群（訓練注意力機制）

    Args:
        size: 3D體積大小（計算用）
        img_size: 最終投影影像大小（用於計算球體尺寸）
        n_clusters: 群組數量（None = 隨機 2-3 群）
        n_objects_per_cluster: 每群球體數量（None = 隨機 2-4 個）

    Returns:
        volume: [D, H, W] 3D array (值越大 = 密度越高 = 吸收越多)
        all_spheres: List of all sphere parameters [(cx, cy, cz, radius, density), ...]
        target_spheres: List of target cluster spheres (用於 bbox)
        target_cluster_id: 目標群組的 ID
    """
    volume = np.zeros((size, size, size), dtype=np.float32)

    if n_clusters is None:
        n_clusters = 3  # 2-3 群

    # 創建座標網格
    z, y, x = np.ogrid[:size, :size, :size]

    # 轉換比例
    scale = size / img_size
    min_radius = int(25 * scale / 2)
    max_radius = int(45 * scale / 2)

    # 為每個群組分配位置（確保群組之間有足夠距離）
    cluster_centers = []
    min_cluster_distance = size // 6  # 群組間最小距離

    for i in range(n_clusters):
        while True:
            # 在體積中隨機選擇群組中心（避免太靠近邊緣）
            margin = size // 8
            cx = np.random.randint(2 * margin, size - 2 * margin)
            cy = np.random.randint(margin, size - margin)
            cz = np.random.randint(margin, size - margin)

            # 檢查與已有群組的距離
            if i == 0:
                cluster_centers.append((cx, cy, cz))
                break
            else:
                distances = [cy-c[1] for c in cluster_centers]
                if min(distances) >= min_cluster_distance:
                    cluster_centers.append((cx, cy, cz))
                    break

    # 為每個群組生成球體
    all_spheres = []
    cluster_spheres_list = []  # 記錄每個群組的球體

    for cluster_id, (center_x, center_y, center_z) in enumerate(cluster_centers):
        if n_objects_per_cluster is None:
            n_objs = np.random.randint(1, 4)  # 每群 2-4 個球體
        else:
            n_objs = n_objects_per_cluster

        cluster_spheres = []
        first_radius = None

        for i in range(n_objs):
            radius = np.random.randint(min_radius, max_radius)

            if i == 0:
                # 第一個球體在群組中心
                cx, cy, cz = center_x, center_y, center_z
                first_radius = radius
            else:
                # 其他球體與第一個球體稍微接觸（不要重疊太多）
                # 距離在 (r1+r2-2) 到 (r1+r2) 之間：稍微接觸到剛好分離
                min_distance = first_radius + radius - 2  # 稍微接觸
                max_distance = first_radius + radius      # 剛好分離
                distance = np.random.uniform(min_distance, max_distance)

                # 隨機方向
                theta = np.random.uniform(0, 2*np.pi)
                phi = np.random.uniform(0, np.pi)

                cx = center_x + int(distance * np.sin(phi) * np.cos(theta))
                cy = center_y + int(distance * np.sin(phi) * np.sin(theta))
                cz = center_z + int(distance * np.cos(phi))

            # 密度
            density = np.random.uniform(0.015, 0.05)

            # 球體
            dist = np.sqrt((x - cx)**2 + (y - cy)**2 + (z - cz)**2)
            mask = dist <= radius
            volume[mask] = np.minimum(volume[mask] + density, 0.15)

            sphere = (cx, cy, cz, radius, density)
            all_spheres.append(sphere)
            cluster_spheres.append(sphere)

        cluster_spheres_list.append(cluster_spheres)

    # 隨機選擇一個群組作為目標（用於 bbox）
    target_cluster_id = np.random.randint(0, n_clusters)
    target_spheres = cluster_spheres_list[target_cluster_id]

    return volume, all_spheres, target_spheres, target_cluster_id


def xray_projection(volume, angle_deg):
    """生成符合 X-ray 物理的投影

    X-ray 強度衰減: I = I0 * exp(-μ * d)
    其中 μ 是吸收係數，d 是穿透距離

    Args:
        volume: [D, H, W] 3D array (密度/吸收係數)
        angle_deg: 投影角度

    Returns:
        projection: [H, W] 2D projection (高值=亮=沒吸收，低值=暗=吸收多)
    """
    # 旋轉體積
    rotated = rotate(volume, angle_deg, axes=(0, 2), reshape=False, order=1)

    # 沿 Z 軸計算累積吸收（線積分）
    # 累積密度
    absorption = np.sum(rotated, axis=0)

    # Beer-Lambert law: I = I0 * exp(-μ * absorption)
    # I0 = 1.0 (最大強度)
    # 縮放因子降低，避免物體全黑（物體應該有部分光透過）
    I0 = 1.0
    projection = I0 * np.exp(-absorption * 2.0)  # 增加對比但保證有光透過

    # 結果：背景（沒物體）= 1.0（亮），物體部分 < 1.0（暗）

    return projection


def add_realistic_xray_noise(projection, noise_level=0.02):
    """加入真實的 X-ray 噪聲

    - Poisson noise (photon counting noise)
    - Electronic noise (Gaussian)
    - Detector non-uniformity
    """
    # Poisson noise (X-ray 的主要噪聲來源)
    # 模擬光子計數：強度高 → 噪聲小，強度低 → 噪聲相對大
    photon_count = 1000  # 平均光子數
    projection_photons = projection * photon_count
    projection_noisy = np.random.poisson(projection_photons) / photon_count

    # Gaussian noise (電子噪聲)
    gaussian_noise = np.random.normal(0, noise_level, projection.shape)
    projection_noisy = projection_noisy + gaussian_noise

    # Detector non-uniformity (探測器不均勻性) - 二維版本
    # 每個像素的靈敏度略有不同
    # 使用二維隨機場模擬真實探測器的空間不均勻性
    sensitivity = 1.0 + np.random.uniform(-0.05, 0.05, projection.shape)
    projection_noisy = projection_noisy * sensitivity

    # Ring artifacts (常見於 CT)
    if np.random.random() < 0.3:  # 30% 機率出現
        ring_intensity = np.random.uniform(-0.02, 0.02, projection.shape[1])
        projection_noisy = projection_noisy + ring_intensity[np.newaxis, :]

    # Clip to valid range
    projection_noisy = np.clip(projection_noisy, 0, 1)

    return projection_noisy


def compute_bbox_from_spheres(spheres, angle_deg, vol_size, img_size):
    """從3D球體精確計算2D投影bbox

    對於球體中心(cx, cy, cz)，半徑r，繞Y軸旋轉θ後投影：
    - 投影後X範圍：球心旋轉後的x ± r（球體投影仍是圓形，半徑不變）
    - 投影後Y範圍：cy ± r（Y軸旋轉不影響Y坐標）

    注意：scipy.ndimage.rotate 的旋轉方向與標準數學定義相反，需要取負角度

    Args:
        spheres: List of (cx, cy, cz, radius, density)
        angle_deg: 投影角度（繞Y軸旋轉）
        vol_size: 3D體積大小
        img_size: 投影影像大小

    Returns:
        bbox: [x1, y1, x2, y2] 精確框住所有球體的bbox
    """
    # scipy.ndimage.rotate 的旋轉方向相反，所以取負角度
    angle_rad = np.deg2rad(-angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)

    all_x_min = []
    all_x_max = []
    all_y_min = []
    all_y_max = []

    scale = img_size / vol_size

    for cx, cy, cz, radius, density in spheres:
        # 轉換到以中心為原點的座標系
        cx_centered = cx - vol_size / 2
        cy_centered = cy - vol_size / 2
        cz_centered = cz - vol_size / 2

        # 球心繞Y軸旋轉後的位置
        cx_rot = cx_centered * cos_a - cz_centered * sin_a
        cy_rot = cy_centered  # Y軸不變

        # 關鍵：球體投影後仍是圓形，半徑為r
        # 所以投影bbox就是：(cx_rot ± r, cy_rot ± r)
        all_x_min.append(cx_rot - radius)
        all_x_max.append(cx_rot + radius)
        all_y_min.append(cy_rot - radius)
        all_y_max.append(cy_rot + radius)

    # 所有球體的聯合bbox
    bbox_x_min = np.min(all_x_min)
    bbox_x_max = np.max(all_x_max)
    bbox_y_min = np.min(all_y_min)
    bbox_y_max = np.max(all_y_max)

    # 轉換到影像座標系統（原點在左上角）
    bbox_x_min_img = int((bbox_x_min + vol_size / 2) * scale)
    bbox_x_max_img = int((bbox_x_max + vol_size / 2) * scale)
    bbox_y_min_img = int((bbox_y_min + vol_size / 2) * scale)
    bbox_y_max_img = int((bbox_y_max + vol_size / 2) * scale)

    # 確保在影像範圍內
    bbox_x_min_img = max(0, bbox_x_min_img)
    bbox_x_max_img = min(img_size - 1, bbox_x_max_img)
    bbox_y_min_img = max(0, bbox_y_min_img)
    bbox_y_max_img = min(img_size - 1, bbox_y_max_img)

    return [bbox_x_min_img, bbox_y_min_img, bbox_x_max_img, bbox_y_max_img]


def create_cvat_xml(output_dir, name, n_frames, bboxes, img_size=512):
    """創建 CVAT 格式的 XML 標註"""
    root = ET.Element("annotations")

    meta = ET.SubElement(root, "meta")
    task = ET.SubElement(meta, "task")
    ET.SubElement(task, "name").text = name
    ET.SubElement(task, "size").text = str(n_frames)

    for frame_id in sorted(bboxes.keys()):
        bbox = bboxes[frame_id]

        image = ET.SubElement(root, "image")
        image.set("id", str(frame_id))
        image.set("name", f"proj_{frame_id:04d}.tif")
        image.set("width", str(img_size))
        image.set("height", str(img_size))

        box = ET.SubElement(image, "box")
        box.set("label", "object")
        box.set("xtl", f"{bbox[0]:.2f}")
        box.set("ytl", f"{bbox[1]:.2f}")
        box.set("xbr", f"{bbox[2]:.2f}")
        box.set("ybr", f"{bbox[3]:.2f}")

    xml_str = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")

    xml_path = os.path.join(output_dir, "annotations.xml")
    with open(xml_path, 'w') as f:
        f.write(xml_str)

    return xml_path


def generate_one_sequence_clustered(output_dir, seq_name, n_clusters=None, n_objects_per_cluster=None):
    """生成一個序列（多群球體，bbox 只框其中一群）"""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n生成序列: {seq_name}")

    angles = np.linspace(-75, 75, 151)
    vol_size = 128
    img_size = 256

    # 創建多群 3D 體積
    print("  創建多群 3D 體積...")
    volume, all_spheres, target_spheres, target_cluster_id = create_clustered_3d_volume(
        size=vol_size, img_size=img_size,
        n_clusters=n_clusters,
        n_objects_per_cluster=n_objects_per_cluster
    )

    print(f"  ✓ 共 {len(all_spheres)} 個球體，分為多群")
    print(f"  ✓ 目標群組 #{target_cluster_id}，包含 {len(target_spheres)} 個球體")

    # 生成投影
    bboxes = {}

    for i, angle in enumerate(tqdm(angles, desc="  生成投影")):
        # X-ray 投影（包含所有球體）
        projection = xray_projection(volume, angle)

        # 加入噪聲
        projection = add_realistic_xray_noise(projection, noise_level=0.02)

        # Resize
        projection_resized = cv2.resize(projection, (img_size, img_size))

        # 轉換為 uint16
        projection_uint16 = (projection_resized * 65535).astype(np.uint16)

        # 保存
        img_path = os.path.join(output_dir, f"proj_{i:04d}.tif")
        cv2.imwrite(img_path, projection_uint16)

        # 只框目標群組的 bbox
        bbox = compute_bbox_from_spheres(target_spheres, angle, vol_size, img_size)
        bboxes[i] = bbox

    # 創建 XML
    xml_path = create_cvat_xml(output_dir, seq_name, len(angles), bboxes, img_size)

    print(f"  ✓ 完成: {len(bboxes)}/{len(angles)} 幀已標註")
    print(f"  ✓ XML: {xml_path}")

    return len(bboxes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='生成 X-ray Tomography 合成數據')
    parser.add_argument('--mode', type=str, default='clustered',
                       choices=['simple', 'clustered'],
                       help='生成模式：simple=所有球體擠在一起，clustered=多群球體')
    parser.add_argument('--n_objects', type=int, default=3,
                       help='simple模式：球體數量')
    parser.add_argument('--n_clusters', type=int, default=None,
                       help='clustered模式：群組數量（None=隨機2-3群）')
    parser.add_argument('--n_per_cluster', type=int, default=None,
                       help='clustered模式：每群球體數量（None=隨機2-4個）')
    parser.add_argument('--output', type=str, default='../tomo_synthetic',
                       help='輸出目錄')

    args = parser.parse_args()

    output_base = args.output

    print("="*60)
    print("生成 X-ray Tomography 測試數據")
    print("="*60)
    print(f"模式: {args.mode}")
    print("="*60)

    seq_dir = os.path.join(output_base, "test_clustered_xray")
    n_annotated = generate_one_sequence_clustered(
        seq_dir, "test_clustered_xray",
        n_clusters=args.n_clusters,
        n_objects_per_cluster=args.n_per_cluster
    )

    print("\n" + "="*60)
    print(f"✓ 測試序列已生成 (clustered): {seq_dir}")
    print(f"  包含 151 張投影，{n_annotated} 幀有標註")
    print(f"  多群球體，bbox 只框其中一群（訓練注意力機制）")
    print("="*60)

    print("\n請檢查生成的影像是否符合預期")
    print("如果滿意，可以批量生成更多序列")
