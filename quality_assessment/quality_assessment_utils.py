import numpy as np
from PIL import Image

def psnr(img1, img2):
    """Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.

    Args:
        img1 (PIL.Image or np.ndarray): First image.
        img2 (PIL.Image or np.ndarray): Second image.

    Returns:
        float: PSNR value in decibels (dB).
    """
    img1 = np.array(img1).astype(np.uint8)
    img2 = np.array(img2).astype(np.uint8)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')  # No difference between images

    max_pixel = 255.0
    psnr_value = 20 * np.log10((max_pixel) / np.sqrt(mse))
    return psnr_value

def ssim(img1, img2):
    """Calculate the Structural Similarity Index (SSIM) between two images.

    Args:
        img1 (PIL.Image or np.ndarray): First image.
        img2 (PIL.Image or np.ndarray): Second image.

    Returns:
        float: SSIM value between -1 and 1.
    """

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    C3 = C2 / 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1 = np.sqrt(np.var(img1))
    sigma2 = np.sqrt(np.var(img2))
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0][1]

    luminance = (2*mu1*mu2 + C1)/(mu1**2 + mu2**2 + C1)
    contrast = (2*(sigma1)*(sigma2) + C2)/((sigma1**2) + (sigma2**2) + C2)
    structure = (sigma12 + C3)/(sigma1*sigma2 + C3)

    ssim_value = luminance * contrast * structure
    return ssim_value

if __name__ == "__main__":

    text_file_loc = "D:\\DIP Project\\Super-resolution-downstream\\quality_assessment\\results.txt"
    write_file = open(text_file_loc, 'w') 
    outputs_prefix = "D:\\DIP Project\\outputs"
    HR_prefix = "D:\\DIP Project\\grayscale_HR_images"
    tasks = ['Bicubic', 'IBP', 'SRCNN', 'self_local']
    # tasks = []
    subdirs = ['Set5', 'Set14', 'BSD100']

    for task in tasks:
        for subdir in subdirs:
            output_dir = f"{outputs_prefix}\\{task}\\{subdir}"
            HR_dir = f"{HR_prefix}\\{subdir}"

            psnr_values = []
            ssim_values = []

            import os
            output_images = sorted(os.listdir(output_dir))
            HR_images = [name[:14] + 'HR.png' for name in output_images]
            for img_name, hr_img_name in zip(output_images, HR_images):
                output_img_path = os.path.join(output_dir, img_name)
                HR_img_path = os.path.join(HR_dir, hr_img_name)

                output_img = Image.open(output_img_path).convert('L')
                HR_img = Image.open(HR_img_path).convert('L')

                output_img_np = np.array(output_img)
                HR_img_np = np.array(HR_img)

                psnr_value = psnr(output_img_np, HR_img_np)
                ssim_value = ssim(output_img_np, HR_img_np)

                psnr_values.append(psnr_value)
                ssim_values.append(ssim_value)

            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            write_file.write(f"Task: {task}, Subdir: {subdir}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}\n")
            print(f"Task: {task}, Subdir: {subdir}, Average PSNR: {avg_psnr:.2f} dB, Average SSIM: {avg_ssim:.4f}")



