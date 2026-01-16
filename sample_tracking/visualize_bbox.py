"""可視化bbox確認正確性"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import os

def load_annotations(xml_path):
    """讀取CVAT XML標註"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    annotations = {}
    for image in root.findall('image'):
        frame_id = int(image.get('id'))
        img_name = image.get('name')

        box = image.find('box')
        if box is not None:
            bbox = [
                float(box.get('xtl')),
                float(box.get('ytl')),
                float(box.get('xbr')),
                float(box.get('ybr'))
            ]
            annotations[frame_id] = {'name': img_name, 'bbox': bbox}

    return annotations

def visualize_bboxes(data_dir, xml_path, output_path, frames_to_show=None):
    """可視化選定幀的bbox"""
    annotations = load_annotations(xml_path)

    if frames_to_show is None:
        # 顯示關鍵角度：-90°, -45°, 0°, 45°, 90°
        frames_to_show = [0, 45, 90, 135, 180]

    n_frames = len(frames_to_show)
    fig, axes = plt.subplots(1, n_frames, figsize=(4*n_frames, 4))

    if n_frames == 1:
        axes = [axes]

    for idx, frame_id in enumerate(frames_to_show):
        if frame_id not in annotations:
            continue

        anno = annotations[frame_id]
        img_path = os.path.join(data_dir, anno['name'])

        # 讀取影像
        img = Image.open(img_path)
        img_array = np.array(img)

        # 正規化顯示
        img_display = (img_array - img_array.min()) / (img_array.max() - img_array.min())

        # 顯示影像
        axes[idx].imshow(img_display, cmap='gray')

        # 繪製bbox
        bbox = anno['bbox']
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1

        from matplotlib.patches import Rectangle
        rect = Rectangle((x1, y1), width, height,
                         linewidth=2, edgecolor='lime', facecolor='none')
        axes[idx].add_patch(rect)

        # 計算角度
        angle = -90 + frame_id * 1.0  # 181幀對應-90到90度
        axes[idx].set_title(f'Frame {frame_id}\nAngle: {angle:.1f}°\nBBox: ({int(x1)}, {int(y1)}, {int(x2)}, {int(y2)})')
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"可視化結果已保存: {output_path}")

    # 同時顯示bbox統計
    print("\nBBox 統計:")
    for frame_id in frames_to_show:
        if frame_id in annotations:
            bbox = annotations[frame_id]['bbox']
            angle = -90 + frame_id * 1.0
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            print(f"  Frame {frame_id:3d} ({angle:6.1f}°): bbox=({int(bbox[0]):3d}, {int(bbox[1]):3d}, {int(bbox[2]):3d}, {int(bbox[3]):3d}) size=({int(width):3d}x{int(height):3d})")

if __name__ == "__main__":
    data_dir = "../tomo_synthetic/test_clustered_xray"
    xml_path = os.path.join(data_dir, "annotations.xml")
    output_path = "./temp/bbox_visualization.png"

    os.makedirs("./temp", exist_ok=True)

    # 可視化關鍵幀
    visualize_bboxes(data_dir, xml_path, output_path, frames_to_show=[0, 45, 90, 135, 180])
