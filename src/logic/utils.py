import numpy as np


def norm_to_8bit(img: np.ndarray, clip_lower=0.1, clip_upper=0.1, clip_percent=None, inverse=False):
    """
    以百分位裁切方式將影像正規化為 8 位元。

    Args:
        img: 輸入影像陣列
        clip_lower: 下限裁切百分比（最暗像素）
        clip_upper: 上限裁切百分比（最亮像素）
        clip_percent: 舊參數（僅裁切上限）
        inverse: 是否反相

    Returns:
        8 位元正規化影像
    """
    # 相容舊參數：若提供 clip_percent，僅套用上限裁切。
    if clip_percent is not None:
        clip_upper = clip_percent
        clip_lower = 0.0

    vmin = np.percentile(img, clip_lower)
    vmax = np.percentile(img, 100 - clip_upper)

    # 避免除以零。
    if vmax == vmin:
        vmax = vmin + 1e-7

    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img, 0, 1)

    if inverse:
        img = 1 - img

    img = (img * 255).astype(np.uint8)
    return img


def split_mosaic(img: np.ndarray, rows: int, cols: int):
    if img.ndim == 2:
        img = img[None, ...]

    N, H_total, W_total = img.shape
    H_patch = H_total // rows
    W_patch = W_total // cols

    patches = []

    for frame in img:
        for i in range(rows):
            for j in range(cols):
                patch = frame[
                    i * H_patch : (i + 1) * H_patch,
                    j * W_patch : (j + 1) * W_patch
                ]
                patches.append(patch)

    return np.stack(patches)


def mosaic_stitching(patches: np.ndarray, rows: int, cols: int):
    N, H, W = patches.shape
    assert N % (rows * cols) == 0, "區塊數量無法剛好組成拼接影像"

    mosaic = np.zeros((rows * H, cols * W), dtype=patches.dtype)

    for i in range(rows):
        for j in range(cols):
            patch_idx = i * cols + j
            patch = patches[patch_idx]
            row_idx = rows - 1 - i  # 從下往上
            mosaic[row_idx*H:(row_idx+1)*H, j*W:(j+1)*W] = patch

    return mosaic


def find_duplicate_angles(thetas: np.ndarray):
    """
    回傳重複角度的索引集合（例如 [[idx1, idx2], [idx3, idx4, idx5], ...]）。
    """
    from collections import defaultdict
    bucket = defaultdict(list)
    for idx, t in enumerate(thetas):
        bucket[t].append(idx)

    duplicates = [v for v in bucket.values() if len(v) > 1]
    return duplicates


def angle_sort(images: np.ndarray, thetas: np.ndarray):
    """
    根據角度對影像和角度進行排序。
    """
    sorted_indices = np.argsort(thetas)
    sorted_images = images[sorted_indices]
    sorted_thetas = thetas[sorted_indices]
    return sorted_images, sorted_thetas


def image_resize(img, size):
    from PIL import Image
    img = Image.fromarray(img)
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img)
