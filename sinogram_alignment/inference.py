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


def inference_alignment(model_path='checkpoints/alignment_ep1000.pt', 
                        data_path='D:/Datasets/TXM_Sino/tomo4-E-b2-60s-181p_sino/sino_0210.tif'
                        , sample_id=20, max_shift=50, n_iter=1, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('configs/alignment_self_att_v3.yml', 'r') as f:
        configs = yaml.safe_load(f)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    if os.path.isfile(data_path):
        raw_sino = cv2.imread(data_path, cv2.IMREAD_GRAYSCALE)
        raw_sino = min_max_normalize(raw_sino, 99.9)
    else:
        sino_folders = glob.glob(f'{data_path}/*')
        sino_files = [glob.glob(f'{f}/*tif') for f in sino_folders]
        raw_sino = cv2.imread(sino_files[sample_id], cv2.IMREAD_GRAYSCALE) / 255

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
            
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 4.5))
    gs = axs[0, -1].get_gridspec()
    for ax in axs[:, -1]:
        ax.remove()

    # select specific angles
    # true_shifts = true_shifts[:n_projs]
    total_shift = total_shift[:n_projs]

    # plot ground truth
    recon_temp = recon_fbp_astra(raw_sino)
    axs[0, 0].set_title('Ground Truth', weight='bold')
    axs[0, 0].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 0].axis('off')

    # plot the FBP reconstruction of raw sinogram
    recon_temp = recon_fbp_astra(raw_sino)
    axs[0, 1].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 1].set_title('Raw', weight='bold')
    axs[0, 1].axis('off')

    # plot the corrected reconstruction
    recon_temp = recon_fbp_astra(pred_sino)
    axs[0, 2].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 2].set_title('Corrected', weight='bold')
    axs[0, 2].axis('off')

    # plot true sinogram
    axs[1, 0].imshow(raw_sino, cmap='gray', interpolation='none')
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

    # plot the error
    # axbig = fig.add_subplot(gs[:, -1])
    # axbig.set_title('Alignment Error', weight='bold')
    # axbig.scatter(true_shifts, -total_shift, c='tab:red', s=6, alpha=0.7)
    # axbig.plot([-50, 50], [-50, 50], ls='--', c='black')
    # axbig.set_xlabel('ground truth (pixel)')
    # axbig.set_ylabel('prediction (pixel)')
    # mae = np.mean(np.abs(true_shifts+total_shift))
    # axbig.text(x=1, y=-16, s='MAE=%.1f'%(mae))
    # axbig.set_aspect('equal')

    fig.tight_layout()
    plt.savefig('temp/infer_result.jpeg', bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    inference_alignment()