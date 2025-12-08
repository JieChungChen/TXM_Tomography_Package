import numpy as np
import yaml, random
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
from preprocess import Sinogram_Data_Random_Shift
from model import Alignment_Net
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
from radon import sino_to_slice


def inference_alignment(model_path='checkpoints/alignment_ep250.pt', sample_id=444, degree=180, repeat=1, seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open('configs/alignment_self_att_v3.yml', 'r') as f:
        configs = yaml.safe_load(f)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    dataset = Sinogram_Data_Random_Shift('data_train/real_sino')
    # dataset = Sinogram_Data_Random_Shift('data_test/real_sino')
    true_sino, test_data, true_shifts, mask = dataset.get_full_sino(sample_id, shifted=True, apply_mask=False)
    true_shifts = (true_shifts * 25).astype(int)

    # create key self-attention key padding mask for sparse angle
    start_angle, end_angle = (180-degree)//2, (180+degree)//2+1

    model = Alignment_Net(configs['model_settings']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    new_sino = test_data.copy()
    with torch.no_grad():
        sino = torch.from_numpy(new_sino).unsqueeze(0).float().to(device)
        for r in range(repeat):
            sino = torch.from_numpy(new_sino).unsqueeze(0).float().to(device)
            pred_shift, pred_sino = model(sino, mask=torch.from_numpy(mask).unsqueeze(0).to(device))
            pred_shift, pred_sino = pred_shift.cpu().detach().squeeze().numpy(), pred_sino.cpu().detach().squeeze().numpy()
            print((pred_shift*25).astype(int))

            if r == 0:
                total_shift = -pred_shift *25
            else:
                total_shift = total_shift - pred_shift * 25

            new_sino = pred_sino
            
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 4.5))
    gs = axs[0, -1].get_gridspec()
    for ax in axs[:, -1]:
        ax.remove()

    # select specific angles
    test_data = test_data[start_angle:end_angle]
    new_sino = new_sino[start_angle:end_angle]
    true_shifts = true_shifts[~mask]
    total_shift = total_shift[~mask]

    # plot ground truth
    angles = np.linspace(0, np.pi, degree+1)
    recon_temp = sino_to_slice(true_sino, angles)
    axs[0, 0].set_title('Ground Truth', weight='bold')
    axs[0, 0].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 0].axis('off')

    # plot the FBP reconstruction of raw sinogram
    recon_temp = sino_to_slice(test_data, angles)
    axs[0, 1].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 1].set_title('Raw', weight='bold')
    axs[0, 1].axis('off')

    # plot the corrected reconstruction
    recon_temp = sino_to_slice(new_sino, angles)
    axs[0, 2].imshow(recon_temp, cmap='gray', interpolation='none')
    axs[0, 2].set_title('Corrected', weight='bold')
    axs[0, 2].axis('off')

    # plot true sinogram
    axs[1, 0].imshow(true_sino, cmap='gray', interpolation='none')
    axs[1, 0].set_xlabel('position (pixel)')
    axs[1, 0].set_ylabel('degree')

    # plot the shifted sinogram
    axs[1, 1].imshow(test_data, cmap='gray', interpolation='none')
    axs[1, 1].set_xlabel('position (pixel)')
    axs[1, 1].set_ylabel('degree')

    # plot the corrected sinogram
    axs[1, 2].imshow(new_sino, cmap='gray', interpolation='none')
    axs[1, 2].set_xlabel('position (pixel)')
    axs[1, 2].set_ylabel('degree')

    # plot the error
    axbig = fig.add_subplot(gs[:, -1])
    axbig.set_title('Alignment Error', weight='bold')
    axbig.scatter(true_shifts, -total_shift, c='tab:red', s=6, alpha=0.7)
    axbig.plot([-50, 50], [-50, 50], ls='--', c='black')
    axbig.set_xlabel('ground truth (pixel)')
    axbig.set_ylabel('prediction (pixel)')
    mae = np.mean(np.abs(true_shifts+total_shift))
    axbig.text(x=1, y=-16, s='MAE=%.1f'%(mae))
    axbig.set_aspect('equal')

    fig.tight_layout()
    plt.savefig('figures/infer_result.jpeg', bbox_inches='tight')
    plt.close()


if __name__=='__main__':
    inference_alignment()