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

        # Load HR and LR
        hr = Image.open(hr_path).convert("L")
        lr = hr.resize((hr.width // self.scale, hr.height // self.scale), Image.BICUBIC)

        W, H = hr.size
        ps = self.patch_size

        # Random HR patch with alignment
        x = random.randint(0, W - ps - 1)
        y = random.randint(0, H - ps - 1)
        x -= x % self.scale
        y -= y % self.scale

        hr_patch = hr.crop((x, y, x + ps, y + ps))

        # Matching LR patch
        lr_ps = ps // self.scale
        lr_x = x // self.scale
        lr_y = y // self.scale

        lr_patch = lr.crop((lr_x, lr_y, lr_x + lr_ps, lr_y + lr_ps))

        hr_patch = np.array(hr_patch).astype(np.float32)/255.0

        # Upsample LR → HR size
        lr_up = lr_patch.resize((ps, ps), Image.BICUBIC)
        lr_up = np.array(lr_up).astype(np.float32)/255.0
        return TF.to_tensor(lr_up), TF.to_tensor(hr_patch)
    
