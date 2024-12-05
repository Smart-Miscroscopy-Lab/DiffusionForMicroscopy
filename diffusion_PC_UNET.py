
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
from torch import nn
import math
import os
from torchvision.utils import save_image



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
BATCH_SIZE = 2

def load_transformed_dataset():
    data_transforms = [
        transforms.Resize((IMG_SIZE,IMG_SIZE)), #crop
        transforms.RandomVerticalFlip(p=0.5),
        transforms.Pad(padding = int(0.2*IMG_SIZE), padding_mode = 'reflect'), #reflect boundary to get rid of borders
        transforms.RandomRotation((-40,40)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.CenterCrop((IMG_SIZE, IMG_SIZE)), #crop back to original size
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1) # Scale between [-1, 1]
    ]
    data_transform = transforms.Compose(data_transforms)

    train = Dataset.ImageFolder(root= '/users/gpb21161/Grant/Datasets/GaussianBlurPC', transform=data_transform)

    test = Dataset.ImageFolder(root= '/users/gpb21161/Grant/Datasets/GaussianBlurPC', transform=data_transform)
    return torch.utils.data.ConcatDataset([train, test])


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



class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_filters=64):
        super(UNet, self).__init__()
        self.in_conv = DoubleConv(in_channels, base_filters)

        # Time embedding layers
        self.time_embedding = nn.Sequential(
            nn.Linear(1, base_filters * 4),
            nn.ReLU(),
            nn.Linear(base_filters * 4, base_filters * 8),
            nn.ReLU(),
        )

        self.down1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(base_filters, base_filters * 2)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(base_filters * 2, base_filters * 4)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(base_filters * 4, base_filters * 8)
        )
        self.bottleneck = DoubleConv(base_filters * 8, base_filters * 16)

        self.up3 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(base_filters * 16, base_filters * 8)

        self.up2 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(base_filters * 8, base_filters * 4)

        self.up1 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(base_filters * 4, base_filters * 2)

        self.out_conv = nn.Conv2d(base_filters * 2, out_channels, kernel_size=1)

    def forward(self, x, t):
        # Encode timestep t into a feature map
        t_emb = self.time_embedding(t.unsqueeze(-1).float())  # Ensure t has the right shape
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)  # Reshape for broadcasting

        # Encoder
        enc1 = self.in_conv(x)
        enc2 = self.down1(enc1)
        enc3 = self.down2(enc2)
        enc4 = self.down3(enc3)

        # Bottleneck with time embedding added
        bottleneck = self.bottleneck(enc4 + t_emb)

        # Decoder
        up3 = self.up3(bottleneck)
        enc4_resized = F.interpolate(enc4, size=up3.shape[2:], mode="bilinear", align_corners=False)
        dec3 = self.dec3(torch.cat([up3, enc4_resized], dim=1))
        
        up2 = self.up2(dec3)
        enc3_resized = F.interpolate(enc3, size=up2.shape[2:], mode="bilinear", align_corners=False)
        dec2 = self.dec2(torch.cat([up2, enc3_resized], dim=1))

        up1 = self.up1(dec2)
        enc2_resized = F.interpolate(enc2, size=up1.shape[2:], mode="bilinear", align_corners=False)
        dec1 = self.dec1(torch.cat([up1, enc2_resized], dim=1))

        out = self.out_conv(dec1)
        return out




model = UNet(in_channels=3, out_channels=3, base_filters=32)

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
writer = SummaryWriter(log_dir="runs_PC_UNET/diffusion_model_experiment")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.001)
epochs = 100000


# Directories for saving final images
final_images_dir = "saved_images_PC_UNET/final"
os.makedirs(final_images_dir, exist_ok=True)

# Training loop with saving only final generated images
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()

        # Select random timesteps
        t = torch.randint(0, T, (BATCH_SIZE,), device=device).long()

        # Forward diffusion and loss computation
        x_noisy, noise = forward_diffusion_sample(batch[0], t, device=device)
        noise_pred = model(x_noisy, t)
        loss = F.l1_loss(noise, noise_pred)  # Use consistent loss calculation
        loss.backward()
        optimizer.step()

        # Log loss
        writer.add_scalar("Loss/train", loss.item(), epoch * len(dataloader) + step)

    # Save final generated image at specified epochs
    if epoch % 100 == 0:  # Adjust frequency as needed
        img = torch.randn((1, 3, IMG_SIZE, IMG_SIZE), device=device)
        for i in range(T - 1, -1, -1):  # Reverse timesteps
            t_sample = torch.full((1,), i, device=device, dtype=torch.long)
            img = sample_timestep(img, t_sample)
            img = torch.clamp(img, -1.0, 1.0)

        # Save final image in .tiff format
        save_image(img, f"{final_images_dir}/epoch_{epoch:03d}_final.tiff", normalize=True, range=(-1, 1))

        print(f"Epoch {epoch}: Final image saved.")

writer.close()


