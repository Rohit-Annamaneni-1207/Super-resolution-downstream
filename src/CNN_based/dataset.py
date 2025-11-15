# from torch.utils.data import Dataset
# import os
# from PIL import Image

# def HR_set_list():
#     file_name_list = []
#     image_list = []
#     for root, dirs, files in os.walk("D:\DIP Project\Train\DIV2K_train_HR"):
#         for file in files:
#             if file.endswith(".jpg") or file.endswith(".png"):
#                 file_name_list.append(os.path.join(root, file))
#                 image = Image.open(os.path.join(root, file))
#                 image_list.append(image)
#     return file_name_list, image_list

# def LR_set_list(HR_file_name):
#     file_name_list = []
#     image_list = []
#     for file_name in HR_file_name:
#         LR_name = file_name[:-4]+'x2.png'
#         file_name_list.append(LR_name)

#     LR_dir = "D:\DIP Project\Train\DIV2K_train_LR_bicubic\X2"
#     for file in file_name_list:
#         image = Image.open(os.path.join(LR_dir, file))
#         image_list.append(image)
#     return file_name_list, image_list


import os
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import numpy as np
import random
import cv2

def get_random_crop(image, crop_height, crop_width):
    # Ensure the image is larger than the crop size
    if image.shape[0] < crop_height or image.shape[1] < crop_width:
        raise ValueError("Image size is smaller than the target crop size")

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    # Randomly select the starting top-left corner (x, y)
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # Slice the image using the selected coordinates and target dimensions
    cropped = image[y: y + crop_height, x: x + crop_width]
    return cropped

def get_centre_crop(image_array, crop_width, crop_height):
    """
    Performs a center crop on a NumPy image array.

    Args:
        image_array (np.ndarray): The input image as a NumPy array.
                                  Expected shape: (height, width, channels) or (height, width).
        crop_width (int): The desired width of the cropped image.
        crop_height (int): The desired height of the cropped image.

    Returns:
        np.ndarray: The center-cropped image array.
    """
    original_height, original_width = image_array.shape[:2]

    # Calculate starting coordinates for the crop
    start_x = (original_width - crop_width) // 2
    start_y = (original_height - crop_height) // 2

    # Calculate ending coordinates for the crop
    end_x = start_x + crop_width
    end_y = start_y + crop_height

    # Perform the crop using NumPy slicing
    cropped_image = image_array[start_y:end_y, start_x:end_x]

    return cropped_image


class DIV2KPatchDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale=3, patch_size=48, patches_per_image=10, mode: str='train'):
        
        self.hr_paths = sorted(list(Path(hr_dir).glob("*.png")) +
                               list(Path(hr_dir).glob("*.jpg")))
        self.lr_dir = Path(lr_dir)
        self.scale = scale
        self.patch_size = patch_size
        self.patches_per_image = patches_per_image
        self.mode = mode

    def __len__(self):
        # Each HR image yields N patches → dataset length is multiplied
        return len(self.hr_paths) * self.patches_per_image

    def __getitem__(self, idx):
        # Select which HR image to sample from
        img_index = idx // self.patches_per_image
        hr_path = self.hr_paths[img_index]

        # Load HR image
        hr = np.array(Image.open(hr_path).convert("L")).astype(np.float32)/255.0

        if self.mode == 'train':
            hr = get_random_crop(hr, self.patch_size, self.patch_size)
        else:
            hr = get_centre_crop(hr, self.patch_size, self.patch_size)

        lr = cv2.resize(hr, (hr.shape[1] // self.scale, hr.shape[0] // self.scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)
        lr = cv2.resize(lr, (lr.shape[1] * self.scale, lr.shape[0] * self.scale), interpolation=cv2.INTER_CUBIC).astype(np.float32)

        # # Build LR path (0001.png → 0001x2.png)
        # base = hr_path.stem
        # lr_path = self.lr_dir / f"{base}x{self.scale}.png"
        # lr = Image.open(lr_path).convert("L")

        # # --- Random HR patch ---
        # ps = self.patch_size
        # x = random.randint(0, W - ps - 1)
        # y = random.randint(0, H - ps - 1)
        # hr_patch = hr.crop((x, y, x + ps, y + ps))

        # # --- Corresponding LR patch ---
        # lr_ps = ps // self.scale
        # lr_x = x // self.scale
        # lr_y = y // self.scale

        # lr_patch = lr.crop((lr_x, lr_y, lr_x + lr_ps, lr_y + lr_ps))

        # # --- Bicubic upsampling of LR patch ---
        # lr_up_patch = lr_patch.resize((ps, ps), resample=Image.BICUBIC)

        # Convert to tensors [1,H,W]
        # return TF.to_tensor(lr_up_patch), TF.to_tensor(hr_patch)
        return TF.to_tensor(lr), TF.to_tensor(hr)
    
