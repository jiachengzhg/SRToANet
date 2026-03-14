import torch
import torch.optim as optim
import numpy as np
import os
from models.waveform_net import WaveformNet, calculate_reward

def train_waveform_net(n_subcarriers=128, n_epochs=1000, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training WaveformNet on {device}")
    
    net = WaveformNet(n_subcarriers).to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    
    # Frequency grid (normalized to -1 to 1)
    freq_grid = torch.linspace(-1, 1, n_subcarriers).to(device)
    
    for epoch in range(n_epochs):
        # Generate random channel states for training
        # In reality, you'd use H from your CIR_Generation logic
        h_sq = torch.exp(torch.randn(batch_size, n_subcarriers) * 0.5).to(device)
        sigma_sq = torch.ones(batch_size, n_subcarriers).to(device) * 0.01
        alpha = torch.rand(batch_size, 1).to(device)
        
        optimizer.zero_grad()
        p = net(h_sq, sigma_sq, alpha)
        
        reward, r_comm, r_sensing = calculate_reward(p, h_sq, sigma_sq, alpha, freq_grid)
        loss = -reward.mean() # Maximize reward
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Comm: {r_comm.mean().item():.2f}, Sens: {r_sensing.mean().item():.4f}")
            
    # Save the model
    save_dir = 'experiments/waveform_design'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(net.state_dict(), os.path.join(save_dir, 'waveform_net.w'))
    print(f"Model saved to {os.path.join(save_dir, 'waveform_net.w')}")

if __name__ == "__main__":
    train_waveform_net()

