import argparse
import os
import yaml
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from preprocess import Sinogram_Data_Random_Shift
from generator import sinogram_mae_build
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
    os.makedirs(trn_conf['model_save_dir'], exist_ok=True)

    model = sinogram_mae_build(configs['model_settings']).to(device)
    if trn_conf['ckpt_path'] is not None:
        model.load_state_dict(torch.load(trn_conf['model_save_dir']+'/'+trn_conf['ckpt_path'], map_location=device))
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
                pred_sino = model(shifted_sino)
                loss = criterion(pred_sino, sino_gt)

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
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].imshow(pred_sino[5].cpu().detach().numpy(), cmap='gray')
            ax[1].imshow(sino_gt[5].cpu().detach().numpy(), cmap='gray')
            plt.savefig('sino_align_gen/temp/sino_epoch_'+str(epoch+1)+'.jpeg')
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