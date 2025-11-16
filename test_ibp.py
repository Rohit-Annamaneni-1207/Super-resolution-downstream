from PIL import Image
from src.iterative_backprojection import IBP
from src.Self_Similarity.res.python.self_similarity import super_res_self_sim
import numpy as np
import cv2
import os

# output_dir = "D:\\DIP Project\\outputs\\IBP"
# os.makedirs(output_dir, exist_ok=True)
# prefix = "D:\\DIP Project\\grayscale_LR_images"
# subdirs = ["BSD100", "Set5", "Set14"]
# suffix = "image_SRF_3"

# for subdir in subdirs:
#     test_dir = os.path.join(prefix, subdir)
#     image_paths = sorted(os.listdir(test_dir))

#     os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

#     for image_path in image_paths:
#         print(f"IBP {image_path}...")

#         lr_image = Image.open(os.path.join(test_dir, image_path)).convert("L")
#         # lr_image = np.array(lr_image.resize((lr_image.width * 3, lr_image.height * 3), Image.BICUBIC)).astype(np.float32)/255.0
        
#         lr_image = np.array(lr_image).astype(np.float32)

#         sr_image = IBP(lr_image, factor_x=3, factor_y=3)

#         cv2.imwrite(os.path.join(output_dir, subdir, image_path), sr_image)

output_dir = "D:\\DIP Project\\outputs\\self_local"
os.makedirs(output_dir, exist_ok=True)
prefix = "D:\\DIP Project\\grayscale_LR_images"
subdirs = ["BSD100", "Set5", "Set14"]
suffix = "image_SRF_3"

for subdir in subdirs:
    test_dir = os.path.join(prefix, subdir)
    image_paths = sorted(os.listdir(test_dir))

    os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    for image_path in image_paths:
        print(f"Self similarity {image_path}...")

        lr_image = Image.open(os.path.join(test_dir, image_path)).convert("RGB")
        # lr_image = np.array(lr_image.resize((lr_image.width * 3, lr_image.height * 3), Image.BICUBIC)).astype(np.float32)/255.0
        
        lr_image = np.array(lr_image).astype(np.float32)/255.0
        sr_image = super_res_self_sim(lr_image, s=3)

        sr_image = Image.fromarray(np.astype(sr_image*255.0, np.uint8)).convert("L")
        sr_image = np.array(sr_image)

        # sr_image = IBP(lr_image, factor_x=3, factor_y=3)
        # sr_image = cv2.resize(lr_image, (lr_image.shape[1]*3, lr_image.shape[0]*3), interpolation=cv2.INTER_CUBIC)

        cv2.imwrite(os.path.join(output_dir, subdir, image_path), sr_image)
