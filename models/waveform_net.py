import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WaveformNet(nn.Module):
    def __init__(self, n_subcarriers):
        super(WaveformNet, self).__init__()
        self.n_subcarriers = n_subcarriers
        
        self.fc = nn.Sequential(
            nn.Linear(n_subcarriers * 2 + 1, 256),  # Input: |H|^2, sigma^2, alpha
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_subcarriers),
            nn.Softmax(dim=-1) # Power allocation sum to 1
        )
        
    def forward(self, h_sq, sigma_sq, alpha):
        """
        h_sq: (batch, n_subcarriers) - Channel power gains
        sigma_sq: (batch, n_subcarriers) - Noise power
        alpha: (batch, 1) - Trade-off factor (0: comm, 1: sensing)
        """
        x = torch.cat([h_sq, sigma_sq, alpha], dim=-1) # when inferring, [1, 257]
        p = self.fc(x)
        return p

def calculate_reward(p, h_sq, sigma_sq, alpha, freq_grid):
    """
    p: (batch, n_subcarriers)
    h_sq: (batch, n_subcarriers)
    sigma_sq: (batch, n_subcarriers)
    alpha: (batch, 1)
    freq_grid: (n_subcarriers,)
    """
    # Communication Reward (Capacity)
    snr = (p * h_sq) / sigma_sq
    r_comm = torch.log2(1 + snr).sum(dim=-1, keepdim=True)
    
    # Sensing Reward (RMS Bandwidth / Gabor Bandwidth)
    # Normalize freq_grid to [-1, 1] for stability
    f_mean = (p * freq_grid).sum(dim=-1, keepdim=True)
    f_var = (p * (freq_grid - f_mean)**2).sum(dim=-1, keepdim=True)
    r_sensing = f_var # RMS bandwidth proxy
    
    # Normalize rewards to similar scales for training stability
    # Communication: Average bits per subcarrier (typically 0.5 - 2.0)
    r_comm_norm = r_comm / h_sq.shape[-1]
    
    # Sensing: Variance is between 0.33 (uniform) and 1.0 (ideal sensing)
    r_sensing_norm = r_sensing
    
    reward = (1 - alpha) * r_comm_norm + alpha * r_sensing_norm
    return reward, r_comm, r_sensing

