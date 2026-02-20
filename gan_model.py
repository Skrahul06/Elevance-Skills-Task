import torch
import torch.nn as nn
from attention import SelfAttention  # <--- IMPORT THE NEW FILE

class Generator(nn.Module):
    def __init__(self, embedding_dim=768, noise_dim=100, image_channels=3):
        super(Generator, self).__init__()
        self.input_dim = noise_dim + embedding_dim
        
        # Initial Block
        self.l1 = nn.Sequential(
            nn.Linear(self.input_dim, 256 * 4 * 4),
            nn.BatchNorm1d(256 * 4 * 4),
            nn.ReLU(True),
            nn.Unflatten(1, (256, 4, 4))
        )
        
        # Upsampling Block 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), 
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )
        
        # --- ATTENTION LAYER ADDED HERE (Task 2 Requirement) ---
        self.attn1 = SelfAttention(128)
        # -------------------------------------------------------
        
        # Upsampling Block 2
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
        )
        
        # Final Output Block
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, image_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, noise, text_embedding):
        combined_input = torch.cat((noise, text_embedding), dim=1)
        
        x = self.l1(combined_input)
        x = self.up1(x)
        
        # Apply Attention
        x, _ = self.attn1(x)  
        
        x = self.up2(x)
        x = self.final(x)
        return x