import argparse, sys, os, yaml
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from preprocess import Sinogram_Data_Random_Shift
from model import sino_align_transformer_builder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from recon_algorithms import recon_fbp_astra


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
    scaler = GradScaler() if device.type == 'cuda' else None
    os.makedirs(trn_conf['model_save_dir'], exist_ok=True)

    model = sino_align_transformer_builder(configs['model_settings']).to(device)
    if trn_conf['ckpt_path'] is not None:
        model.load_state_dict(torch.load(trn_conf['model_save_dir']+'/'+trn_conf['ckpt_path'], map_location=device))

    dataset = Sinogram_Data_Random_Shift(trn_conf['trn_data_dir'], data_conf['max_shift'])
    dataloader = DataLoader(dataset, batch_size=trn_conf['batch_size'], num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    
    criterion_shift = nn.HuberLoss() # L1 seems to be better than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=trn_conf['base_lr'], weight_decay=trn_conf['weight_decay'])
    model.train()
    
    loss_profile = []
    for epoch in range(trn_conf['n_epochs']):
        step = 0
        loss_shift_accum, loss_sino_accum = 0, 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for sino_gt, shifted_sino, shifts, mask in tqdmDataLoader:
                sino_gt = sino_gt.float().to(device) # (batch_size, num_projections, detector_pixels)
                shifted_sino = shifted_sino.float().to(device) # (batch_size, num_projections, detector_pixels)
                shifts = shifts.unsqueeze(2).float().to(device) # (batch_size, num_projections, 1)
                mask = mask.to(device) # (batch_size, num_projections)

                with autocast(device.type):
                    pred_shifts, pred_sino = model(shifted_sino)
                    loss_shift = criterion_shift(pred_shifts[~mask], shifts[~mask])
                    # angle_mask = (~mask).unsqueeze(-1).float() 
                    # diff = (pred_sino - sino_gt) * angle_mask
                    # loss_sino = (diff ** 2).sum() / angle_mask.sum()

                    loss = loss_shift

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf['grad_clip'])
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                step+=1
                loss_shift_accum += loss_shift.item()
                # loss_sino_accum += loss_sino.item()

                if step%1==0:
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": epoch,
                        "loss_shift: ": loss_shift_accum/step,
                        "loss_sino: ": loss_sino_accum/step,
                    })
            loss_profile.append([loss_shift_accum/step, loss_sino_accum/step])
        
        if (epoch+1)%trn_conf['visualize_n_epochs'] == 0:
            fig, ax = plt.subplots(2, 2, figsize=(8, 4))
            pred, gt = pred_sino[5].squeeze().cpu().detach().numpy(), sino_gt[5].squeeze().cpu().detach().numpy()
            ax[0, 0].imshow(pred, cmap='gray')
            ax[0, 1].imshow(gt, cmap='gray')
            pred_recon = recon_fbp_astra(pred, angle_interval=1.0, norm=True)
            gt_recon = recon_fbp_astra(gt, angle_interval=1.0, norm=True)
            ax[1, 0].imshow(pred_recon, cmap='gray')
            ax[1, 1].imshow(gt_recon, cmap='gray')
            plt.savefig(f'temp/recon_test_ep{epoch+1}.jpeg')
            plt.close()

        if (epoch+1)%trn_conf['save_n_epochs'] == 0:
            torch.save(model.state_dict(), os.path.join(trn_conf['model_save_dir'], f'alignment_ep{epoch+1}.pt'))    


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