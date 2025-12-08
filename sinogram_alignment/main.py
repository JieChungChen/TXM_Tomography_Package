import argparse, os, yaml
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from preprocess import Sinogram_Data_Random_Shift
from model import Alignment_Net, FBPLoss, TVLoss, shift_range_penalty
from utils import loss_plot


def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--configs', default='configs/alignment_self_att_v3.yml', type=str)
    return parser


def main(args):
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)

    trn_conf = configs['training_settings']
    data_conf = configs['data_settings']
    device = torch.device(trn_conf['device'] if torch.cuda.is_available() else 'cpu')
    os.makedirs(trn_conf['model_save_dir'], exist_ok=True)

    model = Alignment_Net(configs['model_settings']).to(device)
    if trn_conf['ckpt_path'] is not None:
        model.load_state_dict(torch.load(trn_conf['model_save_dir']+'/'+trn_conf['ckpt_path'], map_location=device))
    dataset = Sinogram_Data_Random_Shift(trn_conf['trn_data_dir'], data_conf['max_shift'])
    dataloader = DataLoader(dataset, batch_size=trn_conf['batch_size'], num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    criterion_shift = nn.HuberLoss() # L1 seems to be better than MSE
    criterion_recon = FBPLoss(img_size=512).to(device)
    criterion_sino = TVLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=trn_conf['base_lr'], weight_decay=trn_conf['weight_decay'])
    model.train()
    
    loss_profile = []
    angles = torch.linspace(0, torch.pi, 181).to(device)
    for epoch in range(trn_conf['n_epochs']):
        step = 0
        loss_shift_accum, loss_recon_accum, loss_tv_accum = 0, 0, 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for sino_gt, shifted_sino, shifts, mask in tqdmDataLoader:
                sino_gt = sino_gt.float().to(device)
                shifted_sino = shifted_sino.float().to(device)
                shifts = shifts.unsqueeze(2).float().to(device)

                pred_shifts, pred_sino = model(shifted_sino, mask.to(device))
                # shift_penality = shift_range_penalty(pred_shifts)
                loss_shift = criterion_shift(pred_shifts[~mask], shifts[~mask])
                loss_tv = criterion_sino(pred_sino[: ,:, 150:-150])
                loss_recon = criterion_recon(pred_sino, sino_gt, angles)

                loss = 0.1*loss_shift + loss_recon + loss_tv
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf['grad_clip'])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step+=1
                loss_shift_accum += loss_shift.item()
                loss_recon_accum += loss_recon.item()
                loss_tv_accum += loss_tv.item()

                if step%1==0:
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": epoch,
                        "loss_shift: ": loss_shift_accum/step,
                        "loss_recon: ": loss_recon_accum/step,
                        "loss_tv: ": loss_tv_accum/step,
                    })
            loss_profile.append([loss_shift_accum/step, loss_recon_accum/step, loss_tv_accum/step])
            pred_recon, raw_recon = criterion_recon.get_fbp_results(pred_sino, sino_gt, angles, circle=False)
            fig, ax = plt.subplots(2, 2, figsize=(8, 4))
            ax[0, 0].imshow(pred_sino[5].cpu().detach().numpy(), cmap='gray')
            ax[0, 1].imshow(sino_gt[5].cpu().detach().numpy(), cmap='gray')
            ax[1, 0].imshow(pred_recon[5].cpu().detach().numpy(), cmap='gray')
            ax[1, 1].imshow(raw_recon[5].cpu().detach().numpy(), cmap='gray')
            plt.savefig('recon_test.jpeg')
            plt.close()

        if (epoch+1)%trn_conf['save_n_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(trn_conf['model_save_dir'], f'alignment_ep{epoch+1}.pt'))
            loss_plot(loss_profile)
    


if __name__=='__main__':
    args = get_args_parser().parse_args()

    if args.train:
        main(args)
    else:
        data = Sinogram_Data_Random_Shift()
        sino, shift, mask = data[0]
        plt.imshow(data.get_full_sino(0), cmap='gray')
        plt.axis('off')
        plt.show()
        plt.close()