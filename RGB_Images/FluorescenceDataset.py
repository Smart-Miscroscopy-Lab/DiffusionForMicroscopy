import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class FluorescenceDataset(Dataset):
    def __init__(self, root_dir, image_size, desired_padding):
        self.root_dir = root_dir
        self.class_folders = [os.path.join(root_dir, class_name) for class_name in os.listdir(root_dir)]
        self.image_size = image_size
        self.desired_padding = desired_padding

        #rezie images into specific cahnnel then convert to a tensor
        self.transform_channel = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        #stack the channels then apply transforms to actual iamges
        self.post_stack_transform = transforms.Compose([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Pad(padding=desired_padding, padding_mode='reflect'),
            transforms.RandomRotation((-40, 40)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.CenterCrop((image_size, image_size)),
            #transforms.Normalize([0.5], [0.5]),  # Normalize grayscale
        ])

    def __len__(self):
        return len(self.class_folders)

    def __getitem__(self, idx):
        class_folder = self.class_folders[idx]
        images = sorted(os.listdir(class_folder))  # make sire in same orders

        # Images are organised into green: red: blue
        green_img = Image.open(os.path.join(class_folder, images[0]))
        red_img = Image.open(os.path.join(class_folder, images[1]))
        blue_img = Image.open(os.path.join(class_folder, images[2]))

        # Apply transformations and stack images
        green_tensor = self.transform_channel(green_img)
        red_tensor = self.transform_channel(red_img)
        blue_tensor = self.transform_channel(blue_img)
        
        # Stack into a single 3D tensor: [3, H, W]
        stacked_img = torch.stack([green_tensor, red_tensor, blue_tensor], dim = 1)

        #stacked_img = self.clamp_intensity(stacked_img)

        
        # Apply post-stack transforms
        stacked_img = self.post_stack_transform(stacked_img)

        stacked_img = stacked_img.squeeze() #ensure correct shape remove any extra dimentsions

        return stacked_img


