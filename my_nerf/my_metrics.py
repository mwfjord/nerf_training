import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PeakSignalNoiseRatioMO(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, y):
        mse = torch.mean((x - y) ** 2)
        psnr = 10 * torch.log10(1 / mse)
        return psnr

class StructuralSimilarityIndexMeasureMO(nn.Module):
    def __init__(self):
        super().__init__()
        self.window_size = 11
        self.sigma = 1.5
        
        dynamic_range = 1
        self.C1 = (0.01 * dynamic_range) ** 2
        self.C2 = (0.03 * dynamic_range) ** 2
    
    def create_window(self, channel):
        gauss = torch.tensor([math.exp(-(x - self.window_size//2)**2/float(2*self.sigma**2)) for x in range(self.window_size)], device=device)
        gauss = gauss / gauss.sum()
        _1D_window = gauss.unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, self.window_size, self.window_size).contiguous()
        return window

    def forward(self, x, y):
        (_, channel, _, _) = x.size()
        window = self.create_window(channel)
        
        mu_x = F.conv2d(x, window, padding=self.window_size//2, groups=channel)
        mu_y = F.conv2d(y, window, padding=self.window_size//2, groups=channel)
        
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)
        mu_x_mu_y = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(x * x, window, padding=self.window_size//2, groups=channel) - mu_x_sq
        sigma_y_sq = F.conv2d(y * y, window, padding=self.window_size//2, groups=channel) - mu_y_sq
        sigma_xy = F.conv2d(x * y, window, padding=self.window_size//2, groups=channel) - mu_x_mu_y
        
        ssim_map = ((2 * mu_x_mu_y + self.C1) * (2 * sigma_xy + self.C2)) / ((mu_x_sq + mu_y_sq + self.C1) * (sigma_x_sq + sigma_y_sq + self.C2))
        
        return ssim_map.mean()

