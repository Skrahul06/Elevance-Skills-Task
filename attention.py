import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """
    Self-Attention Layer for GANs.
    Allows the model to learn relationships between distant pixels.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.in_channels = in_channels
        
        # 1. Create Query, Key, and Value layers (standard Attention mechanism)
        self.query = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key   = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        
        # 2. Gamma: A learnable parameter that controls how much attention to apply
        # We start with 0 so the model initially relies on standard Conv layers
        self.gamma = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, C, width, height = x.size()
        
        # Project inputs to Query, Key, Value
        proj_query = self.query(x).view(batch_size, -1, width * height).permute(0, 2, 1)
        proj_key   = self.key(x).view(batch_size, -1, width * height)
        energy     = torch.bmm(proj_query, proj_key)
        
        # Generate Attention Map
        attention  = self.softmax(energy)
        
        proj_value = self.value(x).view(batch_size, -1, width * height)
        out        = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out        = out.view(batch_size, C, width, height)
        
        # Add Attention output to original input (Residual connection)
        out = self.gamma * out + x
        return out, attention