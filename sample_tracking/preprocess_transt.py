import cv2, os, glob
import numpy as np
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from tqdm import tqdm


def min_max_norm(img, max_percent=99.5, inverse=False):
    vmax = np.percentile(img, max_percent)
    img = np.clip(img, 0, vmax)
    img = (img - img.min()) / (img.max() - img.min())
    if inverse:
        img = 1 - img
    return img


def from_xml(tomo_path, xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    info = []

    # 讀取所有 frame
    n_proj = len(root.findall("image"))
    for img_tag in root.findall("image"):
        img_id = int(img_tag.get("id"))
        img_name = img_tag.get("name")
        img_name = os.path.basename(img_name)
        img_path = os.path.join(tomo_path, img_name)

        box_tag = img_tag.find("box")
        if box_tag is None:
            continue

        xtl, ytl = float(box_tag.get("xtl")), float(box_tag.get("ytl"))
        xbr, ybr = float(box_tag.get("xbr")), float(box_tag.get("ybr"))
        bbox = [xtl, ytl, xbr, ybr]
        info.append({
            "proj_id": img_id,
            "proj": img_path,
            "n_proj": n_proj,
            "bbox": bbox,
            "sequence": os.path.basename(tomo_path)
        })
    return info


def xyxy_to_cxcywh(bbox):
    """轉換 bbox 從 (x1,y1,x2,y2) 到 (cx,cy,w,h)"""
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    return [cx, cy, w, h]


def normalize_bbox(bbox, img_size):
    """將 bbox 標準化到 [0,1]"""
    cx, cy, w, h = bbox
    return [cx/img_size, cy/img_size, w/img_size, h/img_size]


class TransTTrackingDataset(Dataset):
    def __init__(self,
                 root_dir,
                 template_size=128,
                 search_size=256,
                 mode='train',
                 aug_prob=1,
                 max_shift_ratio=0.9):
        """
        Args:
            root_dir: 資料根目錄
            template_size: template 影像大小
            search_size: search 影像大小
            mode: 'train' 或 'test'
            aug_prob: augmentation 概率 (0-1)
            max_shift_ratio: 最大位移比例，相對於可用空間
        """
        self.root_dir = root_dir
        self.template_size = template_size
        self.search_size = search_size
        self.mode = mode
        self.aug_prob = aug_prob
        self.max_shift_ratio = max_shift_ratio

        # 載入樣本 - 按照原本邏輯
        self.samples = []
        self._load_samples()

    def _load_samples(self):
        tomo_dirs = sorted(glob.glob(f"{self.root_dir}/*"))

        for tomo_path in tqdm(tomo_dirs, desc="Loading samples"):
            xml_path = os.path.join(tomo_path, "annotations.xml")
            if not os.path.exists(xml_path):
                continue

            info_dict = from_xml(tomo_path, xml_path)
            self.samples += info_dict

        print(f"Loaded {len(self.samples)} samples")

    def _adjust_bbox(self, bbox, orig_size):
        """調整 bbox 對應 resize 後影像"""
        h0, w0 = orig_size
        scale_x = self.search_size / w0
        scale_y = self.search_size / h0
        x1, y1, x2, y2 = bbox
        return [x1*scale_x, y1*scale_y, x2*scale_x, y2*scale_y]

    def _crop_and_pad(self, img, bbox, context_margin=0.1):
        """按照原本邏輯：bbox + context 裁剪影像，保持比例，用 padding 補齊，再 resize

        Args:
            img: 輸入影像
            bbox: bounding box [x1, y1, x2, y2]
            context_margin: 周圍背景的比例（相對於bbox大小），預設10%
        """
        h, w = img.shape
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2)/2, (y1 + y2)/2
        img = np.roll(img, shift=(128 - int(cx), 128 - int(cy)), axis=(1, 0))
        bw, bh = (x2 - x1), (y2 - y1)

        # 加入 context margin：擴大裁剪範圍
        # 原本的 size 是 max(bw, bh)，現在乘以 (1 + 2*margin) 來保留周圍背景
        # 2*margin 是因為兩側都要加
        size = int(max(bw, bh) * (1 + 2 * context_margin))

        # 計算 crop 區域
        x1c = int(max(0, 128 - size/2))
        y1c = int(max(0, 128 - size/2))
        x2c = int(min(w, 128 + size/2))
        y2c = int(min(h, 128 + size/2))

        patch = img[y1c:y2c, x1c:x2c]
        patch = cv2.resize(patch, (self.template_size, self.template_size))
        return patch

    def _random_roll_augmentation(self, img, bbox):
        """對 search image 進行隨機 roll augmentation，確保 bbox 不超出邊界

        Args:
            img: search image [H, W]
            bbox: bounding box [x1, y1, x2, y2] in pixel coordinates

        Returns:
            img_rolled: augmented image
            bbox_adjusted: adjusted bounding box
        """
        h, w = img.shape
        x1, y1, x2, y2 = bbox

        # 計算安全的最大 shift 範圍（確保 bbox 不超出 [0, w) 和 [0, h)）
        # 向左最多移動 x1，向右最多移動 (w - x2)
        max_shift_x_left = int(x1)
        max_shift_x_right = int(w - x2)

        # 向上最多移動 y1，向下最多移動 (h - y2)
        max_shift_y_up = int(y1)
        max_shift_y_down = int(h - y2)

        # 應用最大位移比例限制
        max_shift_x_left = int(max_shift_x_left * self.max_shift_ratio)
        max_shift_x_right = int(max_shift_x_right * self.max_shift_ratio)
        max_shift_y_up = int(max_shift_y_up * self.max_shift_ratio)
        max_shift_y_down = int(max_shift_y_down * self.max_shift_ratio)

        # 隨機選擇 shift（左負右正，上負下正）
        shift_x = np.random.randint(-max_shift_x_left, max_shift_x_right + 1) if max_shift_x_left + max_shift_x_right > 0 else 0
        shift_y = np.random.randint(-max_shift_y_up, max_shift_y_down + 1) if max_shift_y_up + max_shift_y_down > 0 else 0

        # Roll image
        img_rolled = np.roll(img, shift=(shift_x, shift_y), axis=(1, 0))

        # 調整 bbox 坐標
        bbox_adjusted = [
            x1 + shift_x,
            y1 + shift_y,
            x2 + shift_x,
            y2 + shift_y
        ]

        return img_rolled, bbox_adjusted

    def __getitem__(self, idx):
        sample = self.samples[idx]
        proj_id = sample["proj_id"]
        n_proj = sample["n_proj"]

        # search region - 完全按照原本邏輯
        img = cv2.imread(sample["proj"], cv2.IMREAD_GRAYSCALE)
        orig_size = img.shape
        img = cv2.resize(img, (self.search_size, self.search_size))
        img = min_max_norm(img)
        bbox = self._adjust_bbox(sample["bbox"], orig_size)

        # Random roll augmentation (只在訓練時)
        if self.mode == 'train' and np.random.random() < self.aug_prob:
            img, bbox = self._random_roll_augmentation(img, bbox)

        # template - 完全按照原本邏輯，但改用隨機選擇或前一幀
        if self.mode == 'train':
            # 訓練時隨機選擇附近幀
            template_offset = np.random.randint(max(-5, -proj_id),
                                              min(6, n_proj - proj_id))
            template_idx = idx + template_offset
            template_idx = max(0, min(len(self.samples) - 1, template_idx))
        else:
            # 測試時使用前一幀
            template_idx = max(0, idx - 1)

        template_sample = self.samples[template_idx]
        t_img = cv2.imread(template_sample["proj"], cv2.IMREAD_GRAYSCALE)
        t_orig_size = t_img.shape
        t_img = cv2.resize(t_img, (self.search_size, self.search_size))
        t_img = min_max_norm(t_img)
        t_bbox = self._adjust_bbox(template_sample["bbox"], t_orig_size)

        # crop patch - 完全按照原本邏輯
        template_patch = self._crop_and_pad(t_img, t_bbox)
        target_patch = img

        # 轉換為 tensor
        template_tensor = torch.from_numpy(template_patch).unsqueeze(0).float()
        target_tensor = torch.from_numpy(target_patch).unsqueeze(0).float()

        # 轉換 bbox 格式為 TransT 需要的 (cx, cy, w, h) 並標準化
        bbox_cxcywh = xyxy_to_cxcywh(bbox)
        bbox_norm = normalize_bbox(bbox_cxcywh, self.search_size)

        return {
            'search_images': target_tensor,
            'template_images': template_tensor,
            'search_anno': torch.tensor(bbox_norm, dtype=torch.float32),
            'template_anno': torch.tensor([0.5, 0.5, 0.8, 0.8], dtype=torch.float32),  # template 中心化
            'seq_name': sample["sequence"],
            'frame_id': sample["proj_id"]
        }

    def __len__(self):
        return len(self.samples)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # 測試資料集
    dataset = TransTTrackingDataset("/home/jackson29/tomo_alignment/tomo_train", mode='train')
    sample = dataset[333]

    search_img = sample['search_images'].squeeze().numpy()
    template_img = sample['template_images'].squeeze().numpy()
    search_bbox = sample['search_anno'].numpy()

    print(f"Search bbox (normalized): {search_bbox}")

    # 反標準化到像素座標
    cx, cy, w, h = search_bbox * 256  # search_size = 256
    x1, y1 = cx - w/2, cy - h/2

    # 可視化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Template
    ax1.imshow(template_img, cmap='gray', vmin=0, vmax=1)
    ax1.set_title('Template (following original logic)')
    ax1.axis('off')

    # Search with bbox
    ax2.imshow(search_img, cmap='gray', vmin=0, vmax=1)
    rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='red', facecolor='none')
    ax2.add_patch(rect)
    ax2.set_title('Search with bbox (following original logic)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()
    plt.close()
    print("Following original preprocess.py logic exactly")

    # folder = '../tomo_train/tomo06-Mur-bulk-1-C22-b4-40s-161p'
    # files = sorted(glob.glob(f'{folder}/*.tif'))
    # for i, f in enumerate(files):
    #     img_temp = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    #     img_temp = min_max_norm(img_temp)
    #     img_temp = (img_temp *255).astype(np.uint8)
    #     cv2.imwrite(f'../tomo_train/tomo06-Mur-bulk-1-C22-b4-40s-161p_norm/{str(i).zfill(3)}.tif', img_temp)