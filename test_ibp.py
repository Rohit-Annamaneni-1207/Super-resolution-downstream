from PIL import Image
from src.iterative_backprojection import IBP
import numpy as np
import cv2
import os

output_dir = "D:\\DIP Project\\outputs\\IBP"
os.makedirs(output_dir, exist_ok=True)
prefix = "D:\\DIP Project\\grayscale_LR_images"
subdirs = ["BSD100", "Set5", "Set14"]
suffix = "image_SRF_3"

for subdir in subdirs:
    test_dir = os.path.join(prefix, subdir)
    image_paths = sorted(os.listdir(test_dir))

    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for image_path in image_paths:
        print(f"IBP {image_path}...")

        lr_image = Image.open(os.path.join(test_dir, image_path)).convert("L")
        # lr_image = np.array(lr_image.resize((lr_image.width * 3, lr_image.height * 3), Image.BICUBIC)).astype(np.float32)/255.0
        
        lr_image = np.array(lr_image).astype(np.float32)

        sr_image = IBP(lr_image, factor_x=3, factor_y=3)

        cv2.imwrite(os.path.join(output_dir, subdir, image_path), sr_image)
