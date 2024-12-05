
# 
# **Sources:**
# - Github implementation [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
# - Niels Rogge, Kashif Rasul, [Huggingface notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023)
# - Papers on Diffusion models ([Dhariwal, Nichol, 2021], [Ho et al., 2020] ect.)
# 



# %%
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
from parallel_nodes import setup_distributed
from torch.nn.parallel import DistributedDataParallel as DDP
import math
import torchvision.models as models


# %%


def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


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

# %% [markdown]
# Let's test it on our dataset ...

# %%


IMG_SIZE = 512
BATCH_SIZE = 28

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE,IMG_SIZE)), #crop
        transforms.RandomVerticalFlip(p=0.5),
        #transforms.Pad(padding = int(0.2*IMG_SIZE), padding_mode = 'reflect'), #reflect boundary to get rid of borders
        #transforms.RandomRotation((-40,40)),
        transforms.RandomHorizontalFlip(p=0.5),
        #transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), #crop back to original size
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = Dataset.ImageFolder(root= '/users/gpb21161/Grant/Datasets/LiveCell/BT474', transform=data_transform)

    return train




data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
# Simulate forward diffusion
image = next(iter(dataloader))[0]

num_images = 10
stepsize = int(T/num_images)




class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()

    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


import torchvision.models as models


class FineTunedUNet(nn.Module):
    def __init__(self, pretrained_encoder=True):
        super().__init__()
        image_channels = 3  # Input images are RGB (adjust for your data if needed)
        time_emb_dim = 32
        out_dim = 3  # Output channels

        # Load ImageNet-pretrained ResNet as the encoder
        resnet = models.resnet34(pretrained=pretrained_encoder)
        self.encoder_blocks = nn.ModuleList([
            nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool),  # Block 1
            resnet.layer1,  # Block 2
            resnet.layer2,  # Block 3
            resnet.layer3,  # Block 4
            resnet.layer4   # Block 5
        ])
        
        # Define the decoder with learnable layers
        decoder_channels = [512, 256, 128, 64, 64]
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_channels[i], decoder_channels[i + 1], time_emb_dim, up=True)
            for i in range(len(decoder_channels) - 1)
        ])

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Initial projection and output layer
        self.input_layer = nn.Conv2d(image_channels, 64, 3, padding=1)
        self.output_layer = nn.Conv2d(64, out_dim, kernel_size=1)
	

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder: downsample the image
        skip_connections = []
        for block in self.encoder_blocks:
            x = block(x)
            skip_connections.append(x)

        # Decoder: upsample and concatenate skip connections
        for block in self.decoder_blocks:
            skip = skip_connections.pop()
            x = torch.cat((x, skip), dim=1)
            x = block(x, t_emb)

        return self.output_layer(x)






# %%
def get_loss(model, x_0, t):
    x_noisy, noise = forward_diffusion_sample(x_0, t, device)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise, noise_pred)


# %%
@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    noise_pred = model(x, t)
    
    # Ensure `noise_pred` matches the size of `x`
    if noise_pred.shape != x.shape:
        noise_pred = F.interpolate(noise_pred, size=x.shape[-2:], mode="bilinear", align_corners=False)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# %% [markdown]
# ## Training

# %%
# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runsPC_SH5Y_pretrained/diffusion_model_experiment")



device = "cuda" if torch.cuda.is_available() else "cpu"
model = FineTunedUNet(pretrained_encoder=True).to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100000


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
# Directories for saving final images and model checkpoints
final_images_dir = "saved_images_PC_SH5Y_pretrained/final"
model_save_dir = "saved_models_PC_SH5Y_pretrained"
os.makedirs(final_images_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

##to start training from checkpoint: # Load checkpoint if exists
checkpoint_path = "saved_models_PC_SH5Y_pretrained/model_epoch_3300.pth"  # Change to your latest checkpoint file
start_epoch = 0

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}...")
else:
    print("No checkpoint found, starting training from scratch.")


for epoch in range(start_epoch, epochs):
    model.train()
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Select random timesteps
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

        # Compute noisy images and ground truth noise
        x_noisy, noise = forward_diffusion_sample(batch[0].to(device), t, device=device)

        # Predict noise using the model
        noise_pred = model(x_noisy, t)

        # Resize `noise_pred` to match the shape of `noise` if they differ
        if noise_pred.shape != noise.shape:
            noise_pred = F.interpolate(noise_pred, size=noise.shape[-2:], mode="bilinear", align_corners=False)

        # Compute loss
        loss = F.l1_loss(noise, noise_pred)
        loss.backward()
        optimizer.step()

        # Log loss
        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)

    # Save final generated image and model checkpoint at specified epochs
    if epoch % 50 == 0:  # Adjust frequency as needed
        # Generate and save the final image
        img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
        for i in range(0, T)[::-1]:
            t_sample = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t_sample)
            img = torch.clamp(img, -1.0, 1.0)

        # Save final image in .tiff format
        save_image(img, f"{final_images_dir}/epoch_{epoch:03d}_final.tiff", normalize=True, range=(-1, 1))
        print(f"Epoch {epoch}: Final image saved.")

        # Save the model checkpoint
        model_save_path = f"{model_save_dir}/model_epoch_{epoch:03d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, model_save_path)
        print(f"Epoch {epoch}: Model checkpoint saved at {model_save_path}")

# Close the TensorBoard writer
writer.close()




