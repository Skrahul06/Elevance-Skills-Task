import torch
import torch.nn as nn

class BasicShapeCGAN(nn.Module):
    def __init__(self, num_classes=2, noise_dim=50, embed_dim=10):
        super(BasicShapeCGAN, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.input_dim = noise_dim + embed_dim
        
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 28 * 28), 
            nn.Tanh() 
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], dim=1)
        img = self.model(x)
        img = img.view(img.size(0), 1, 28, 28)
        return img

def test_shape_cgan():
    print("--- Task 6: Basic Shapes Conditional GAN ---")
    
    generator = BasicShapeCGAN(num_classes=2)
    generator.eval()
    
    print("\nTesting 'Square' Generation...")
    noise = torch.randn(1, 50) 
    label_square = torch.tensor([0]) 
    
    with torch.no_grad():
        generated_shape = generator(noise, label_square)
        
    print(f"Generated Output Shape: {generated_shape.shape} (1 channel, 28x28 pixels)")
    print("\n✅ SUCCESS: Basic Shapes CGAN architecture is working!")

if __name__ == "__main__":
    test_shape_cgan()