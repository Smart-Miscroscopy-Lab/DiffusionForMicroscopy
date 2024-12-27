from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 128,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250    # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
)

import os

dataset_path = '/users/gpb21161/Grant/Datasets/LiveCell/SH5Y/SH5Y'
files = os.listdir(dataset_path)
print(f"Total files: {len(files)}")
print(f"First few files: {files[:10]}")


trainer = Trainer(
    diffusion,
    '/users/gpb21161/Grant/Datasets/LiveCell/SH5Y',
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)



trainer.train()