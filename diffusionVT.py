# %%
from transformers import ViTModel
import torch
import torchvision
import torch.nn as nn
from torch.optim import Adam
import torchvision.datasets as Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision.utils import save_image


# %% Beta schedule and helper functions
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise, noise

# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# %% Data loading and transformations
IMG_SIZE = 224
BATCH_SIZE = 16

def load_transformed_dataset():
    data_transforms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda t: (t * 2) - 1)
    ])
    train = Dataset.ImageFolder(root='/users/gpb21161/Grant/Datasets/LiveCell/BT474', transform=data_transforms)
    return train

data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

class VisionTransformerDiffusionModel(nn.Module):
    def __init__(self, pretrained_model_name="google/vit-base-patch16-224-in21k"):
        super().__init__()
        # Load pretrained ViT
        self.vit = ViTModel.from_pretrained(pretrained_model_name)
        self.hidden_size = self.vit.config.hidden_size
        self.patch_size = self.vit.config.patch_size
        
        # Add time embedding layer
        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        
        # Output projection for noise prediction
        self.output_layer = nn.Linear(self.hidden_size, 3)

    def forward(self, x, t):
        # Input validation
        if len(x.shape) != 4 or x.shape[1] != 3:
            raise ValueError(f"Expected input shape (B, 3, H, W), but got {x.shape}")
        
        batch_size = x.shape[0]
        img_size = x.shape[-1]

        # Pass input image through ViT
        vit_output = self.vit(pixel_values=x).last_hidden_state  # Shape: (B, num_patches, hidden_size)
        
        # Add time embedding
        t_emb = self.time_mlp(t.unsqueeze(-1).float())  # Shape: (B, hidden_size)
        t_emb = t_emb.unsqueeze(1).expand(-1, vit_output.size(1), -1)  # Shape: (B, num_patches, hidden_size)
        x = vit_output + t_emb  # Shape: (B, num_patches, hidden_size)
        
        # Predict noise for each patch
        x = self.output_layer(x)  # Shape: (B, num_patches, 3)
        
        # Reshape patches back to image
        num_patches = int((img_size / self.patch_size) ** 2)
        patch_dim = self.patch_size
        x = x.view(batch_size, int(img_size / patch_dim), int(img_size / patch_dim), 3)  # (B, H/patch_size, W/patch_size, 3)
        x = x.permute(0, 3, 1, 2)  # (B, 3, H/patch_size, W/patch_size)
        x = F.interpolate(x, size=(img_size, img_size), mode='bilinear', align_corners=False)  # Upsample to original image size
        return x


# %% Loss and sampling functions
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    print(f"Noise shape: {noise.shape}, Noise prediction shape: {noise_pred.shape}")  # Debugging
    return F.mse_loss(noise, noise_pred)


@torch.no_grad()
def sample_timestep(x, t):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    noise_pred = model(x, t)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# %% Training loop
writer = SummaryWriter(log_dir="runsPC_BT474_ViT/diffusion_model_experiment")

device = "cuda" if torch.cuda.is_available() else "cpu"
model = VisionTransformerDiffusionModel().to(device)
optimizer = Adam(model.parameters(), lr=0.0001)
epochs = 100000

final_images_dir = "saved_images_PC_BT474_ViT/final"
model_save_dir = "saved_models_PC_BT474_ViT"
os.makedirs(final_images_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

for epoch in range(epochs):
    model.train()
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()
        x_noisy, noise = forward_diffusion_sample(batch[0].to(device), t, device=device)
        noise_pred = model(x_noisy, t)
        loss = get_loss(model, batch[0].to(device), t)
        loss.backward()
        optimizer.step()

        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)

    if epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
            for i in range(0, T)[::-1]:
                t_sample = torch.full((1,), i, device=device, dtype=torch.long)
                img = sample_timestep(img, t_sample)
                img = torch.clamp(img, -1.0, 1.0)
        save_image(img, f"{final_images_dir}/epoch_{epoch:03d}_final.png", normalize=True, range=(-1, 1))
        model_save_path = f"{model_save_dir}/model_epoch_{epoch:03d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, model_save_path)

writer.close()
