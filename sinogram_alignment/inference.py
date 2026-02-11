import sys
import os
import glob
import numpy as np
import cv2
import yaml
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from model import sino_align_transformer_builder
mpl.rcParams['figure.dpi'] = 400
plt.rcParams["xtick.direction"] = 'in'
plt.rcParams["ytick.direction"] = 'in'
params = {
   'axes.labelsize': 10,
   'font.size': 12,
   'legend.fontsize': 10,
   'xtick.labelsize': 10,
   'ytick.labelsize': 10,
   'text.usetex': False,
   'figure.figsize': [4, 4],
   'mathtext.fontset': 'stix',
   'font.family': 'STIXGeneral'
   }
plt.rcParams.update(params)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recon_algorithms import recon_fbp_astra
from utils import min_max_normalize


def apply_shift(sinogram, shifts):
    n_projs, size = sinogram.shape
    for i in range(n_projs):
        sinogram[i] = torch.roll(sinogram[i], shifts[i], dims=0)
    return sinogram


def inference_alignment(model_path='checkpoints/alignment_ep500.pt', 
                        data_path='temp/sino_0294.tif'
                        , sample_id=20, max_shift=50, n_iter=5, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('configs/alignment_self_att_v3.yml', 'r') as f:
        configs = yaml.safe_load(f)

    if os.path.isfile(data_path):
        raw_sino = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        raw_sino = raw_sino.astype(np.float32)
        if raw_sino.shape[1] != 512:
            raw_sino = cv2.resize(raw_sino, (512, raw_sino.shape[0]), interpolation=cv2.INTER_LINEAR)

        for i in range(raw_sino.shape[0]):
            raw_sino[i, :] = min_max_normalize(raw_sino[i, :].copy(), 99.9)
        # raw_sino = min_max_normalize(raw_sino, 99.9)
        raw_sino = 1 - raw_sino
    else:
        sino_folders = glob.glob(f'{data_path}/*')
        sino_files = [glob.glob(f'{f}/*tif') for f in sino_folders]
        raw_sino = cv2.imread(sino_files[sample_id], cv2.IMREAD_GRAYSCALE) / 255

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        
        x = np.arange(len(raw_sino))
        shift = (
            np.random.uniform(0, max_shift*0.5) * np.sin(2 * np.pi * np.random.uniform(0.02, 0.2) * x) +
            np.random.uniform(-max_shift*0.5, max_shift*0.5, size=len(raw_sino))
        ).astype(int)
        gt_sino = raw_sino.copy()
        for i in range(len(raw_sino)):
            raw_sino[i, :] = np.roll(raw_sino[i, :], shift[i])
    else:
        gt_sino = raw_sino.copy()

    
    model = sino_align_transformer_builder(configs['model_settings']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
    model.eval()

    n_projs, detectpr_pixs = raw_sino.shape
    mask = torch.zeros((1, 181), dtype=bool, device=device)
    mask[0, n_projs:] = 1  # mask invalid projections
    sino_temp = torch.zeros((181, detectpr_pixs), dtype=torch.float32)
    sino_temp[:n_projs, :] = torch.from_numpy(raw_sino)

    with torch.no_grad():
        sino_temp = sino_temp.float().to(device)
        
        for i in range(n_iter):
            sino_temp = sino_temp.unsqueeze(0)  # (1, n_projs, detectpr_pixs)
            pred_shift, _ = model(sino_temp, mask)
            pred_shift = pred_shift.cpu().detach().squeeze().numpy()
            pred_shift = (pred_shift * max_shift).astype(int)
            print(pred_shift)

            if i == 0:
                total_shift = -pred_shift
            else:
                total_shift = total_shift - pred_shift

            sino_temp = apply_shift(sino_temp.squeeze(), -pred_shift)
    pred_sino = sino_temp.cpu().detach().squeeze().numpy()
            
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(11, 5))

    # select specific angles
    total_shift = total_shift[:n_projs]

    # plot gt
    if gt_sino is not None:
        recon_temp = recon_fbp_astra(gt_sino, norm=False)
        glob_min, glob_max = recon_temp.min(), recon_temp.max()
        recon_temp = (recon_temp - glob_min) / (glob_max - glob_min)
        hfer_gt = high_freq_energy_ratio(recon_temp)
        axs[0, 0].set_title('Ground Truth', weight='bold')
        axs[0, 0].text(5, 25, f'HFER: {hfer_gt:.1e}', color='yellow', weight='bold', bbox=dict(facecolor='black', alpha=0.5))
        axs[0, 0].imshow(recon_temp, cmap='gray', interpolation='none', vmin=0, vmax=1)
        axs[0, 0].axis('off')

    # plot the FBP reconstruction of raw sinogram
    recon_temp = recon_fbp_astra(raw_sino, norm=False)
    recon_temp = (recon_temp - glob_min) / (glob_max - glob_min)
    hfer_raw = high_freq_energy_ratio(recon_temp)
    axs[0, 1].imshow(recon_temp, cmap='gray', interpolation='none', vmin=0, vmax=1)
    axs[0, 1].text(5, 25, f'HFER: {hfer_raw:.1e}', color='yellow', weight='bold', bbox=dict(facecolor='black', alpha=0.5))
    axs[0, 1].set_title('Raw', weight='bold')
    axs[0, 1].axis('off')

    # plot the corrected reconstruction
    recon_temp = recon_fbp_astra(pred_sino, norm=False)
    recon_temp = (recon_temp - glob_min) / (glob_max - glob_min)
    hfer_cor = high_freq_energy_ratio(recon_temp)
    axs[0, 2].imshow(recon_temp, cmap='gray', interpolation='none', vmin=0, vmax=1)
    axs[0, 2].text(5, 25, f'HFER: {hfer_cor:.1e}', color='yellow', weight='bold', bbox=dict(facecolor='black', alpha=0.5))
    axs[0, 2].set_title('Corrected', weight='bold')
    axs[0, 2].axis('off')

    # plot true sinogram
    if gt_sino is not None:
        axs[1, 0].imshow(gt_sino, cmap='gray', interpolation='none')
        axs[1, 0].set_xlabel('position (pixel)')
        axs[1, 0].set_ylabel('degree')

    # plot the shifted sinogram
    axs[1, 1].imshow(raw_sino, cmap='gray', interpolation='none')
    axs[1, 1].set_xlabel('position (pixel)')
    axs[1, 1].set_ylabel('degree')

    # plot the corrected sinogram
    axs[1, 2].imshow(pred_sino, cmap='gray', interpolation='none')
    axs[1, 2].set_xlabel('position (pixel)')
    axs[1, 2].set_ylabel('degree')

    fig.tight_layout()
    plt.savefig('temp/infer_result.jpeg', bbox_inches='tight')
    plt.close()


def high_freq_energy_ratio(img, f0=0.035):
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=1)
    F = np.fft.fftshift(np.fft.fft2(img))
    power = np.abs(F) ** 2

    H, W = img.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))
    fx = np.fft.fftshift(np.fft.fftfreq(W))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX**2 + FY**2)

    return power[R > f0].sum() / power.sum()


if __name__=='__main__':
    inference_alignment()