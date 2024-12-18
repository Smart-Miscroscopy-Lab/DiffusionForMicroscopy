# %% [markdown]
# # A Diffusion Model from Scratch in Pytorch
# 
# In this notebook I want to build a very simple (as few code as possible) Diffusion Model for generating car images. I will explain all the theoretical details in the YouTube video.
# 
# 
# **Sources:**
# - Github implementation [Denoising Diffusion Pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)
# - Niels Rogge, Kashif Rasul, [Huggingface notebook](https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/annotated_diffusion.ipynb#scrollTo=3a159023)
# - Papers on Diffusion models ([Dhariwal, Nichol, 2021], [Ho et al., 2020] ect.)
# 



# %%
import torch
import torchvision
from torch.optim import Adam
import torchvision.datasets as Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import os



# %% [markdown]
# Later in this notebook we will do some additional modifications to this dataset, for example make the images smaller, convert them to tensors ect.

# %% [markdown]
# # Building the Diffusion Model
# 

# %% [markdown]
# ## Step 1: The forward process = Noise scheduler
# 
# 
# 

# %% [markdown]
# We first need to build the inputs for our model, which are more and more noisy images. Instead of doing this sequentially, we can use the closed form provided in the papers to calculate the image for any of the timesteps individually.
# 
# **Key Takeaways**:
# - The noise-levels/variances can be pre-computed
# - There are different types of variance schedules
# - We can sample each timestep image independently (Sums of Gaussians is also Gaussian)
# - No model is needed in this forward step

# %%
import torch.nn.functional as F

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
T = 1200
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
BATCH_SIZE = 8

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), # Scales data into [0,1]
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = Dataset.ImageFolder(root= '/users/gpb21161/Grant/Datasets/GaussianBlurPC', transform=data_transform)

    return train

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        transforms.Lambda(lambda t: t * 255.),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])


data = load_transformed_dataset()
dataloader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# %%
# Simulate forward diffusion
image = next(iter(dataloader))[0]

num_images = 10
stepsize = int(T/num_images)



# %% [markdown]
# ## Step 2: The backward process = U-Net
# 
# 

# %% [markdown]
# For a great introduction to UNets, have a look at this post: https://amaarora.github.io/2020/09/13/unet.html.
# 
# 
# **Key Takeaways**:
# - We use a simple form of a UNet for to predict the noise in the image
# - The input is a noisy image, the ouput the noise in the image
# - Because the parameters are shared accross time, we need to tell the network in which timestep we are
# - The Timestep is encoded by the transformer Sinusoidal Embedding
# - We output one single value (mean), because the variance is fixed
# 

# %%
from torch import nn
import math


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


class Unet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 3
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = 3
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )

        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # Edit: Corrected a bug found by Jakub C (see YouTube comment)
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)
            x = up(x, t)
        return self.output(x)

model = Unet()
print("Num params: ", sum(p.numel() for p in model.parameters()))
model




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
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


# %% [markdown]
# ## Training

# %%
# Initialize TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs_BT474/diffusion_model_experiment")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100000

checkpoint_path = '/users/gpb21161/Grant/jobs/saved_models_BT474/model_epoch_12900.pth'

if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}...")
else:
    print("No checkpoint found, starting training from scratch.")


import os

# Define directories to save images and models
noisy_images_dir = "saved_images_BT474/noisy"
generated_images_dir = "saved_images_BT474/generated"
model_save_dir = "saved_models_BT474"
os.makedirs(noisy_images_dir, exist_ok=True)
os.makedirs(generated_images_dir, exist_ok=True)
os.makedirs(model_save_dir, exist_ok=True)

# Training loop with image and model saving
for epoch in range(start_epoch,epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Select random timesteps
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

        # Compute loss and backward pass
        x_noisy, noise = forward_diffusion_sample(batch[0].to(device), t, device=device)
        noise_pred = model(x_noisy, t)
        loss = F.l1_loss(noise, noise_pred)
        loss.backward()
        optimizer.step()

        # Log loss
        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)

        # Log and save generated images
        if epoch % 100 == 0 and step == 0:
            sampled_images = []
            img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
            for i in range(0, T)[::-1]:
                t_sample = torch.full((1,), i, device=device, dtype=torch.long)
                img = sample_timestep(img, t_sample)
                img = torch.clamp(img, -1.0, 1.0)
                if i % (T // 10) == 0:  # Save at intervals
                    sampled_images.append(img.clone().detach().cpu())

            grid_generated = torchvision.utils.make_grid(torch.cat(sampled_images, dim=0), normalize=True, range=(-1, 1))
            writer.add_image("Generated Images/Backward Pass", grid_generated, epoch)

        print(f"Epoch {epoch} | Step {step:03d} Loss: {loss.item()} ")

    # Save the model and generated images every 50 epochs
    if epoch % 100 == 0:
        # Save the model checkpoint
        model_save_path = f"{model_save_dir}/model_epoch_{epoch:03d}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, model_save_path)
        print(f"Epoch {epoch}: Model checkpoint saved at {model_save_path}")

        # Save final generated image
        with torch.no_grad():
            img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
            for i in range(0, T)[::-1]:
                t_sample = torch.full((1,), i, device=device, dtype=torch.long)
                img = sample_timestep(img, t_sample)
                img = torch.clamp(img, -1.0, 1.0)

            torchvision.utils.save_image(
                img, f"{generated_images_dir}/epoch_{epoch:03d}_final.png", normalize=True, range=(-1, 1)
            )
            print(f"Epoch {epoch}: Final generated image saved.")

writer.close()




