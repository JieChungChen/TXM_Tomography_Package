import argparse
import os
import yaml
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from preprocess import Sinogram_Data_Random_Shift
from generator import sinogram_mae_build, grad_loss
from tqdm import tqdm


def get_args_parser():
    parser = argparse.ArgumentParser('diffusion for background correction', add_help=False)
    parser.add_argument('--train', default=True, type=bool)
    parser.add_argument('--configs', default='sino_align_gen/configs/sino_align_gen_v1.yml', type=str)
    return parser


def main(args):
    with open(args.configs, 'r') as f:
        configs = yaml.safe_load(f)

    trn_conf = configs['training_settings']
    data_conf = configs['data_settings']
    device = torch.device(trn_conf['device'] if torch.cuda.is_available() else 'cpu')
    scaler = GradScaler() if device.type == 'cuda' else None
    os.makedirs(trn_conf['model_save_dir'], exist_ok=True)

    model = sinogram_mae_build(configs['model_settings']).to(device)
    if trn_conf['ckpt_path'] is not None:
        model.load_state_dict(torch.load(trn_conf['model_save_dir']+'/'+trn_conf['ckpt_path'], map_location=device), strict=False)
    dataset = Sinogram_Data_Random_Shift(trn_conf['trn_data_dir'], data_conf['max_shift'])
    dataloader = DataLoader(dataset, batch_size=trn_conf['batch_size'], num_workers=0, shuffle=True, pin_memory=True, drop_last=True)
    criterion = nn.MSELoss() # L1 seems to be better than MSE

    optimizer = torch.optim.AdamW(model.parameters(), lr=trn_conf['base_lr'], weight_decay=trn_conf['weight_decay'])
    model.train()
    
    for epoch in range(trn_conf['n_epochs']):
        step = 0
        loss_accum = 0
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for sino_gt, shifted_sino in tqdmDataLoader:
                sino_gt = sino_gt.float().to(device)
                shifted_sino = shifted_sino.float().to(device)

                with autocast():
                    pred_sino = model(shifted_sino)
                    mse_loss = criterion(pred_sino, sino_gt)
                    grad_loss_val = grad_loss(pred_sino.unsqueeze(1), sino_gt.unsqueeze(1))
                    loss = mse_loss + 0.2 * grad_loss_val

                if scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # 可选：用于梯度裁剪前的反缩放
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), trn_conf['grad_clip'])
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)
                step+=1
                loss_accum += loss.item()

                if step % 1 == 0:
                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": epoch,
                        "loss: ": loss_accum/step,
                    })

        if (epoch+1) % trn_conf['visualize_n_epochs'] == 0:
            fig, ax = plt.subplots(3, 1, figsize=(8, 9))
            ax[0].imshow(shifted_sino[5].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            ax[0].set_title('Input Shifted')
            ax[1].imshow(pred_sino[5].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            ax[1].set_title('Predicted')
            ax[2].imshow(sino_gt[5].cpu().detach().numpy(), cmap='gray', vmin=0, vmax=1)
            ax[2].set_title('Ground Truth')
            plt.savefig('sino_align_gen/temp/sino_epoch_'+str(epoch+1)+'.jpeg', dpi=300)
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