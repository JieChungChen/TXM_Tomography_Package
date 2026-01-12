import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import glob
import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from TransT_v3 import TransT
from preprocess_transt import min_max_norm
import sys 
sys.path.append("..") 
from sino_alignment.radon import sino_to_slice


def parse_xml(xml_path, inference_size=256):
    """從 XML 解析標註幀和 bbox"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    frames, bboxes = [], {}
    for image in root.findall('image'):
        boxes = image.findall('box')
        if boxes:
            frame_id = int(image.get('id'))
            w, h = int(image.get('width')), int(image.get('height'))
            box = boxes[0]
            bbox = [float(box.get('xtl')), float(box.get('ytl')),
                   float(box.get('xbr')), float(box.get('ybr'))]
            frames.append(frame_id)
            bboxes[frame_id] = bbox

    if not frames:
        raise ValueError("XML 中找不到標註")

    # 選中位數幀
    frames.sort()
    mid = frames[len(frames)//2]
    bbox = bboxes[mid]

    # 縮放 bbox
    scale = inference_size / w
    bbox_scaled = {fid: [x * scale for x in bboxes[fid]] for fid in frames}

    print(f"Template: frame {mid}, bbox {bbox_scaled[mid]}")
    return bbox_scaled[mid], mid


def preprocess_image(img_path, size=256):
    """讀取並預處理影像"""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (size, size))
    img = min_max_norm(img)
    return torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float()


def crop_template(img, bbox, template_size=128, context_margin=0.1):
    """裁剪 template"""
    img_np = img.squeeze().cpu().numpy()
    h, w = img_np.shape
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

    # 移動到中心
    img_shifted = np.roll(img_np, shift=(128 - int(cx), 128 - int(cy)), axis=(1, 0))

    # 裁剪正方形區域
    bw, bh = x2 - x1, y2 - y1
    size = int(max(bw, bh) * (1 + 2 * context_margin))
    x1c = int(max(0, 128 - size/2))
    y1c = int(max(0, 128 - size/2))
    x2c = int(min(w, 128 + size/2))
    y2c = int(min(h, 128 + size/2))

    patch = img_shifted[y1c:y2c, x1c:x2c]
    patch = cv2.resize(patch, (template_size, template_size))

    return torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().to(img.device)


def predict(model, template, search, default_h=None):
    """模型推理 - 使用 top-K 加權平均"""
    with torch.no_grad():
        cls_logits, bbox_pred = model(template, search)

    # 計算 object 信心
    cls_prob = torch.softmax(cls_logits, dim=1)[0, 0]  # [H, W]
    H, W = cls_prob.shape

    # 找 top-K 最高信心位置
    k = min(1, H * W)
    cls_prob_flat = cls_prob.flatten()
    topk_values, topk_indices = torch.topk(cls_prob_flat, k)

    # 提取 top-K 位置的 bbox 預測
    topk_bboxes = []
    for idx in topk_indices:
        y, x = idx // W, idx % W
        topk_bboxes.append(bbox_pred[0, :, y, x])
    topk_bboxes = torch.stack(topk_bboxes)  # [K, 4]

    # 加權平均：使用信心作為權重
    weights = topk_values / topk_values.sum()  # 歸一化權重
    bbox = (topk_bboxes * weights.unsqueeze(1)).sum(dim=0).cpu().numpy()  # [4]
    conf = topk_values[0].item()  # 最大信心值

    # normalized cxcywh -> pixel xyxy
    cx, cy, w, h = bbox * 256

    if default_h is not None:
        h = default_h

    # 回傳 bbox, conf, 以及分類熱度圖
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2], conf, cls_prob.cpu()


def visualize_frame(img, bbox, conf, is_template=False):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    img_np = img.squeeze().numpy()
    ax.imshow(img_np, cmap='gray', vmin=0, vmax=1)
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    # Template frame 用藍色，其他用紅色
    color = 'green' if is_template else 'red'
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    ax.text(x1, y1-5, f'{conf:.3f}', color=color, fontsize=10,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    ax.axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    plt.close()
    return frame


def visualize_frame_with_heatmap(img, bbox, conf, cls_prob, is_template=False):
    # 建立 1x2 子圖
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    img_np = img.squeeze().numpy()
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    color = 'green' if is_template else 'red'

    # === 左：原圖 + bbox ===
    axes[0].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
    axes[0].add_patch(rect)
    axes[0].text(x1, y1-5, f'{conf:.3f}', color=color, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[0].set_title('tracking result', fontsize=12)
    axes[0].axis('off')

    heatmap = cls_prob.numpy()  # [H, W]
    heatmap_upsampled = cv2.resize(heatmap, (256, 256), interpolation=cv2.INTER_LINEAR)

    # === 右：疊加圖 (原圖 + 半透明熱度圖 + bbox) ===
    axes[1].imshow(img_np, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(heatmap_upsampled, cmap='jet', alpha=0.4, vmin=0, vmax=1)
    rect2 = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor=color, facecolor='none')
    axes[1].add_patch(rect2)
    axes[1].text(x1, y1-5, f'{conf:.3f}', color=color, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    axes[1].set_title('cls result', fontsize=12)
    axes[1].axis('off')

    plt.tight_layout()
    fig.canvas.draw()
    frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    plt.close()
    return frame


def track(model, image_paths, bbox, template_idx, device):
    """追蹤序列"""
    print(f"\n從 frame {template_idx} 開始追蹤")

    results = {}
    video_frames = {}
    height = abs(bbox[3] - bbox[1])

    # 準備 template
    template_img = preprocess_image(image_paths[template_idx]).to(device)
    template = crop_template(template_img, bbox)

    # 中間frame直接用ground truth
    results[template_idx] = {'frame': template_idx, 'bbox': bbox, 'conf': 1.0, 'path': image_paths[template_idx]}
    video_frames[template_idx] = visualize_frame(template_img.cpu(), bbox, 1.0, is_template=True)

    # Forward tracking
    print(f"Forward: {template_idx+1} -> {len(image_paths)-1}")
    for i in tqdm(range(template_idx + 1, len(image_paths)), desc="Forward"):
        search = preprocess_image(image_paths[i]).to(device)
        pred_bbox, conf, cls_prob = predict(model, template, search, default_h=height)

        results[i] = {'frame': i, 'bbox': pred_bbox, 'conf': conf, 'path': image_paths[i]}
        video_frames[i] = visualize_frame_with_heatmap(search.cpu(), pred_bbox, conf, cls_prob)

        template = crop_template(search, pred_bbox)

    # Backward tracking
    print(f"Backward: {template_idx-1} -> 0")
    template_img = preprocess_image(image_paths[template_idx]).to(device)
    template = crop_template(template_img, bbox)

    for i in tqdm(range(template_idx - 1, -1, -1), desc="Backward"):
        search = preprocess_image(image_paths[i]).to(device)
        pred_bbox, conf, cls_prob = predict(model, template, search, default_h=height)

        results[i] = {'frame': i, 'bbox': pred_bbox, 'conf': conf, 'path': image_paths[i]}
        video_frames[i] = visualize_frame_with_heatmap(search.cpu(), pred_bbox, conf, cls_prob)

        template = crop_template(search, pred_bbox)

    return [results[i] for i in sorted(results.keys())], video_frames


def create_video(frames, output, fps=10):
    frame_ids = sorted(frames.keys())
    h, w = frames[frame_ids[0]].shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (w, h))

    print(f"\n生成影片: {output}")
    for fid in tqdm(frame_ids, desc="Video"):
        writer.write(frames[fid])
    writer.release()
    return True


def aligned_video(metadata, output, fps=10):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output, fourcc, fps, (256, 256))

    print(f"\n生成影片: {output}")
    mid_sino = []
    for m in tqdm(metadata, desc="Video"):
        img_path = m['path']
        bbox = m['bbox']
        center = ((bbox[0]+bbox[2])//2, (bbox[1]+bbox[3])//2)
        dx, dy = 128 - center[0], 128 - center[1]
        img_temp = preprocess_image(img_path).squeeze().numpy()
        img_temp = np.roll(img_temp, shift=(int(dx), int(dy)), axis=(1, 0))
        mid_sino.append(img_temp[128, :])
        img_temp = (img_temp *255).astype(np.uint8)
        img_temp = cv2.cvtColor(img_temp, cv2.COLOR_GRAY2BGR)
        writer.write(img_temp)
    writer.release()

    mid_sino = np.array(mid_sino)
    angles = np.linspace(0, len(mid_sino)-1, len(mid_sino)) / 180 * np.pi
    recon = sino_to_slice(mid_sino, angles)
    cv2.imwrite('temp/mid_recon.tif', recon)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='checkpoints/transt_ep60.pt')
    parser.add_argument('--data', default='../tomo_test/tomo_simu_04', help='資料夾路徑')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--fps', type=int, default=10)
    args = parser.parse_args()

    # 解析 XML
    xml_path = os.path.join(args.data, 'annotations.xml')
    bbox, template_idx = parse_xml(xml_path)

    # 載入影像
    images = sorted(glob.glob(f"{args.data}/*.tif*"))
    print(f"影像數: {len(images)}")

    # 載入模型
    print(f"模型: {args.model}")
    model = TransT().to(args.device)
    model.load_state_dict(torch.load(args.model, map_location=args.device), strict=True)
    model.eval()

    # 追蹤
    metadata, frames = track(model, images, bbox, template_idx, args.device)

    # 保存結果
    os.makedirs('temp', exist_ok=True)
    name = os.path.basename(args.data.rstrip('/'))

    # with open(f'temp/results_{name}.txt', 'w') as f:
    #     f.write("Frame,X1,Y1,X2,Y2,Conf\n")
    #     for r in metadata:
    #         f.write(f"{r['frame']},{','.join(map(str, r['bbox']))},{r['conf']:.4f}\n")

    video_path = f'temp/result_{name}.mp4'
    aligned_path = f'temp/aligned_{name}.mp4'
    create_video(frames, video_path, args.fps)
    aligned_video(metadata, aligned_path, args.fps)
    print(f"完成: {video_path}")


if __name__ == "__main__":
    main()
