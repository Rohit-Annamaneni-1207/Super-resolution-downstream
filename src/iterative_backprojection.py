import numpy as np
from scipy.ndimage import convolve
import cv2

def gaussian_kernel(size: int=9, sigma: float=1.2):
    kernel = np.zeros((size, size), dtype=np.float32)
    center = size // 2
    for x in range(size):
        for y in range(size):
            kernel[x, y] = np.exp(-((x - center) ** 2 + (y - center) ** 2) / (2 * sigma ** 2))
    return kernel / np.sum(kernel)


def add_gaussian_noise(img:np.array, mean:float=0.0, sigma:float=0.01):

    noise = np.random.normal(mean, sigma, img.shape)*255.0
    noisy_img = img + noise
    noisy_img = np.clip(noisy_img, 0, 255)
    return noisy_img

def decimate(img:np.array, factor_x:int, factor_y:int):

    # decimated_img = img[::factor_x, ::factor_y]
    # return decimated_img
    return cv2.resize(img, None, fx=1/factor_x, fy=1/factor_y, interpolation=cv2.INTER_CUBIC).astype(np.float32)

def degradation(img:np.array, kernel:np.array, factor_x:int, factor_y:int):

    blurred = convolve(img, kernel)
    decimated = decimate(blurred, factor_x, factor_y)
    noisy = add_gaussian_noise(decimated)
    return noisy

def MSE_calc(img1:np.array, img2:np.array):
    
    mse = np.mean((img1 - img2) ** 2)
    return mse

def IBP(lr_img:np.array, factor_x:int, factor_y:int, mse_threshold:float=1e-3, max_iterations:int=100, alpha:float=1.0):
    hr_estimate = cv2.resize(lr_img, None, fx=factor_x, fy=factor_y, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    kernel = gaussian_kernel()
    HBP = np.flipud(np.fliplr(kernel))
    for i in range(max_iterations):
        print(f"Iteration {i+1}")
        # Step 1: Generate LR image from current HR estimate
        lr_generated = degradation(hr_estimate, kernel, factor_x, factor_y)

        # Step 2: Calculate MSE
        mse = MSE_calc(lr_img, lr_generated)

        residual = lr_img - lr_generated
        # print(type(residual))
        # break
        upsampled_residual = cv2.resize(residual, None, fx=factor_x, fy=factor_y, interpolation=cv2.INTER_CUBIC)
        # Step 3: Check for convergence
        if mse < mse_threshold:
            break
        
        # Step 4: Update HR estimate
        hr_estimate += alpha * convolve(upsampled_residual, HBP)
        

    return hr_estimate
