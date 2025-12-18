import cv2
import sys
import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
from generator import sinogram_mae_build, grad_loss
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recon_algorithms import recon_fbp_astra


def main():
    # Load config
    with open('configs/sino_align_gen_v1.yml', 'r') as f:
        configs = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build model
    model = sinogram_mae_build(configs['model_settings']).to(device)
    # Load checkpoint
    checkpoint_path = 'checkpoints/alignment_ep100.pt'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device), strict=True)
    model.eval()

    # Inference on sample data
    file_path = 'temp/sino_0122.tif'
    sino = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    n_projs = sino.shape[0]
    if n_projs < 181:
        sino_temp = np.zeros((181, sino.shape[1]), dtype=sino.dtype)
        sino_temp[:n_projs, :] = sino
        sino = sino_temp
    sino = torch.from_numpy(sino/255).unsqueeze(0).float().to(device)
    with torch.no_grad():
        aligned_sino = model(sino, mask_len=10)

    sino = sino.cpu().squeeze()[:n_projs]
    aligned_sino = aligned_sino.cpu().squeeze()[:n_projs]

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.subplot(2, 1, 1)
    plt.title('Original Sinogram')
    plt.imshow(sino.cpu().squeeze()[:n_projs], cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(2, 1, 2)
    plt.title('Aligned Sinogram')
    plt.imshow(aligned_sino.cpu().squeeze()[:n_projs], cmap='gray', vmin=0, vmax=1)
    plt.xticks([])
    plt.yticks([])
    plt.savefig('inference_result.png', dpi=300)
    plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    recon_raw = recon_fbp_astra(sino.numpy(), angle_interval=1, norm=True)
    ax[0].imshow(recon_raw, cmap='gray')
    ax[0].set_title('recon_raw')
    ax[0].axis('off')
    recon = recon_fbp_astra(aligned_sino.numpy(), angle_interval=1, norm=True)
    ax[1].imshow(recon, cmap='gray')
    ax[1].set_title('recon_aligned')
    ax[1].axis('off')
    fig.tight_layout()
    plt.savefig('recon.png', dpi=300)
    plt.close()


if __name__ == '__main__':
    main()