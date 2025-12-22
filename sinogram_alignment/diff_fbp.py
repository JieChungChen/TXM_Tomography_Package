import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Filter_Back_Projection(nn.Module):
    def __init__(self, img_size, filtered=False):
        super().__init__()
        self.img_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * img_size))))
        self.filtered = filtered
        center = img_size//2
        x, y = torch.meshgrid(
            torch.arange(img_size) - center,
            torch.arange(img_size) - center,
            indexing='ij')
        self.register_buffer("hann", self.get_hann_filter(self.img_size_padded))
        self.register_buffer("x", x)
        self.register_buffer("y", y)
        recon_0 =self.empty_recon(img_size, torch.linspace(0, np.pi, 181))
        self.register_buffer("recon_0", recon_0)
        
    def empty_recon(self, img_size, angles):
        recon = torch.zeros((img_size, img_size))
        sino_empty = torch.ones((1, len(angles), img_size))
        if self.filtered:
            pad_width = (0, self.img_size_padded - img_size)
            sino_padded = torch.nn.functional.pad(sino_empty, pad_width, 'constant', 0)
            sino_fft = torch.fft.fft(sino_padded, dim=-1)
            sino_empty = torch.real(torch.fft.ifft(sino_fft * self.hann, dim=-1))[:, :, :img_size]
        for i, theta in enumerate(angles):
            t = self.x * torch.cos(theta+torch.pi/2) + self.y * torch.sin(theta+torch.pi/2)
            t_idx = torch.round(t + img_size//2).long()  
            valid = (t_idx >= 0) & (t_idx < img_size)
            recon[valid] += sino_empty[0, i, t_idx[valid]]
        return recon
    
    def forward(self, sino, angles, circle=True):
        B, n_proj, L = sino.shape
        center = L//2
        if self.filtered:
            pad_width = (0, self.img_size_padded - L)
            sino_padded = torch.nn.functional.pad(sino, pad_width, 'constant', 0)
            sino_fft = torch.fft.fft(sino_padded, dim=-1)
            sino = torch.real(torch.fft.ifft(sino_fft * self.hann, dim=-1))[:, :, :L]
        
        recon = torch.zeros((B, L, L), device=sino.device)
        for i, theta in enumerate(angles):
            # 計算各theta下reconstruction每個pixel投影在sinogram上的座標
            t = self.x * torch.cos(theta+torch.pi/2) + self.y * torch.sin(theta+torch.pi/2)
            t_norm = 2 * t / (L - 1)  # [-1, 1] for grid_sample

            # grid_sample 要求 shape: [B, 1, H, W]
            t_norm = t_norm.unsqueeze(0).unsqueeze(0).repeat(B, 1, 1, 1)  # [B, 1, H, W]

            # 提取該角度對應的 sinogram
            sino_i = sino[:, i, :].unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]

            # 建立 grid: (x = 投影位置, y = dummy 0)
            grid = torch.stack((t_norm, torch.zeros_like(t_norm)), dim=-1)  # [B, 1, H, W, 2]

            # 使用 grid_sample 把 sinogram 投影值拉回影像座標
            sampled = F.grid_sample(sino_i.float(), grid.squeeze(), mode='bilinear', align_corners=True, padding_mode='zeros')
            recon += sampled[:, 0]

        if circle:
            Y, X = np.ogrid[:L, :L]
            dist_from_center = np.sqrt((X - center)**2 + (Y - center)**2)
            mask = dist_from_center > center
            recon[:, mask] = 0
        else:
            recon /= self.recon_0
        return recon
    
    def get_hann_filter(self, img_size):
        n = torch.concatenate((torch.arange(1, img_size / 2 + 1, 2, dtype=int),
                               torch.arange(img_size / 2 - 1, 0, -2, dtype=int)))
        f = torch.zeros(img_size)
        f[0] = 0.25
        f[1::2] = -1 / (torch.pi * n) ** 2
        ramp = 2 * torch.real(torch.fft.fft(f))
        hann = torch.from_numpy(np.hanning(img_size))
        return ramp * torch.fft.fftshift(hann)
    

class FBPLoss(nn.Module):
    def __init__(self, img_size, filtered=True):
        super().__init__()
        self.fbp = Filter_Back_Projection(img_size, filtered)
        self.mse = nn.MSELoss()

    def forward(self, sino_pred, sino_true, angles):
        recon_pred = self.fbp(sino_pred, angles)
        recon_true = self.fbp(sino_true, angles)

        loss_mse = self.mse(recon_pred, recon_true)
        return loss_mse