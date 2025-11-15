import torch
from PIL import Image
import torchvision.transforms.functional as TF
import os
import numpy as np
import cv2
from train import SRCNN

MODEL_WT_PATH = "D:\\DIP Project\\SRCNN_best_MSE_x3_patched.pth"
MODEL_WT_PATH = "D:\\DIP Project\\srcnn_x3.pth"


def load_model(model, model_wt_path):
    model.load_state_dict(torch.load(model_wt_path))
    model.eval()
    return model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

grayscale_LR_dir = "D:\\DIP Project\\grayscale_LR_images"
grayscale_HR_dir = "D:\\DIP Project\\grayscale_HR_images"
os.makedirs(grayscale_LR_dir, exist_ok=True)
os.makedirs(grayscale_HR_dir, exist_ok=True)


prefix = "D:\\DIP Project\\Test"
subdirs = ["BSD100", "Set5", "Set14"]
suffix = "image_SRF_3"

for subdir in subdirs:
    test_dir = os.path.join(prefix, subdir, suffix)

    image_paths = sorted(os.listdir(test_dir))
    os.makedirs(os.path.join(grayscale_LR_dir, subdir), exist_ok=True)
    os.makedirs(os.path.join(grayscale_HR_dir, subdir), exist_ok=True)

    for image_path in image_paths:
        print(f"Processing {image_path}...")

        image = np.array(Image.open(os.path.join(test_dir, image_path)).convert("L")).astype(np.uint8)

        if "HR" in image_path:
            save_path = os.path.join(grayscale_HR_dir, subdir, image_path)
            cv2.imwrite(save_path, image)
        else:
            save_path = os.path.join(grayscale_LR_dir, subdir, image_path)
            cv2.imwrite(save_path, image)

output_dir = "D:\\DIP Project\\outputs\\Bicubic"
os.makedirs(output_dir, exist_ok=True)
prefix = "D:\\DIP Project\\grayscale_LR_images"
subdirs = ["BSD100", "Set5", "Set14"]

for subdir in subdirs:
    test_dir = os.path.join(prefix, subdir)
    image_paths = sorted(os.listdir(test_dir))

    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for image_path in image_paths:
        print(f"Bicubic upscaling {image_path}...")

        lr_image = Image.open(os.path.join(test_dir, image_path)).convert("L")
        sr_image = lr_image.resize((lr_image.width * 3, lr_image.height * 3), Image.BICUBIC)
        sr_image = np.array(sr_image)

        print("SR Shape",sr_image.shape)
        cv2.imwrite(os.path.join(output_dir, subdir, image_path), sr_image)



output_dir = "D:\\DIP Project\\outputs\\SRCNN"
os.makedirs(output_dir, exist_ok=True)

model = load_model(SRCNN().to(device), MODEL_WT_PATH)

prefix = "D:\\DIP Project\\grayscale_LR_images"
subdirs = ["BSD100", "Set5", "Set14"]

for subdir in subdirs:
    test_dir = os.path.join(prefix, subdir)
    image_paths = sorted(os.listdir(test_dir))

    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for image_path in image_paths:
        print(f"Super-resolving {image_path}...")

        lr_image = Image.open(os.path.join(test_dir, image_path)).convert("L")
        lr_image = np.array(lr_image.resize((lr_image.width * 3, lr_image.height * 3), Image.BICUBIC)).astype(np.float32)/255.0
        lr_tensor = TF.to_tensor(lr_image).unsqueeze(0).to(device)

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        sr_image = np.array(TF.to_pil_image(sr_tensor.squeeze(0)))

        print("SR Shape",sr_image.shape)
        cv2.imwrite(os.path.join(output_dir, subdir, image_path), sr_image)






    

        
        

        







