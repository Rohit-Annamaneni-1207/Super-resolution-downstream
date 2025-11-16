import numpy as np
import cv2
import math
import time

from .transforms import resize, cubic_bspline, quadratic_bspline

def get_search_area(L_0: np.ndarray, i_c, j_c, window_size):
    '''
    **Inputs**
    ----
    - L_0 [np.ndarray]: LR image (can be grayscale or color both)
    - i_c, j_c [int]: center coordinates of the search area
    - window_size [int]: the size of the search area window

    **Output**:
    ----
    - search_patch [np.ndarray]: search area in LR

    **Desciption**
    ----
    Gives the LR search area for example patch  
    '''
    # 1. define ideal boundaries
    radius= window_size//2
    i_s_ideal= i_c - radius
    i_e_ideal= i_c + radius + 1
    j_s_ideal= j_c - radius
    j_e_ideal= j_c + radius + 1

    l0shape= L_0.shape
    i_s_actual= max(0, i_s_ideal)
    i_e_actual= min(l0shape[0], i_e_ideal)
    j_s_actual= max(0, j_s_ideal)
    j_e_actual= min(l0shape[1], j_e_ideal)

    # 2. padding for the thing
    pad_top= i_s_actual - i_s_ideal
    pad_bot= i_e_ideal - i_e_actual
    pad_lft= j_s_actual - j_s_ideal
    pad_rgt= j_e_ideal - j_e_actual

    # 3. extract patch
    search_patch= L_0[i_s_actual: i_e_actual, j_s_actual: j_e_actual]

    # 4. pad patch
    if L_0.ndim > 2:
        search_patch= np.pad(
            search_patch, 
            ((pad_top, pad_bot), (pad_lft, pad_rgt), (0,0)), 
            'constant', constant_values=0
        )
    else:
        search_patch= np.pad(
            search_patch, 
            ((pad_top, pad_bot), (pad_lft, pad_rgt)), 
            'constant', constant_values=0
        )
    
    return search_patch

def get_best_detail_patch(query, search_area_l0, search_area_h0):
    '''
    **Inputs**
    ----
    - query [np.ndarray]: query patch in LR
    - search_area_l0 [np.ndarray]: searching in blurred out version of og image
    - search_area_h0 [int]: HF detail of search area from og image

    **Output**:
    ----
    - h_patch [np.ndarray]: HF detail of most similar patch from search_area_l0 

    **Desciption**
    ----
    Gives the HF detail of most similar patch from search_area_l0
    '''
    assert query.ndim == search_area_h0.ndim and query.ndim == search_area_l0.ndim, "They cannot have different dimensions!"

    result= cv2.matchTemplate(search_area_l0.astype(np.float32), query.astype(np.float32), cv2.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc= cv2.minMaxLoc(result)

    x_tl, y_tl= min_loc
    patch_size_lr= query.shape

    h_patch= search_area_h0[y_tl: y_tl + patch_size_lr[0], x_tl: x_tl + patch_size_lr[1]]

    return h_patch

def process_patch(i_h, j_h, L_1, L_0, H_0, s, patch_size_hr, patch_size_lr, search_window_size):
    '''
    The inner working of a loop that runs parallely now
    '''
    pass

def small_SR(I_0, s: float, patch_size= 7, search_size= 21, overlap= 0.2):
    '''
    **Inputs**
    ----
    - I_0 [np.ndarray]: Og image LR
    - s [float]: small scale factor
    - patch_size [int]: HR image patch size for self similarity
    - search_size[int]: search area size in LR blurred image
    - overlap [float]: overlap percentage of blocks

    **Output**:
    ----
    - img_out [np.ndarray]: upscaled image

    **Desciption**
    ----
    Gives upscaled image
    '''
    assert I_0 is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(I_0,(np.ndarray,)) and len(I_0.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
    assert s > 1, "This is a super resolution routine. Provide proper upscale factor"

    interp_type= 'wavelet'
    stride= round((1. - overlap) * patch_size)
    
    print("Starting LSS Upscale (Strided Method)...")
    start_time = time.time()

    # --- 1. Create image layers --- #
    print("Step 1: Creating Image Layers...")

    # 1.1 L_1 (base HR image)
    H, W, _= I_0.shape
    wu, hu= math.ceil(s*W), math.ceil(s*H)
    L_1= resize(I_0, (hu, wu), interp=interp_type, kernel_func=cubic_bspline)      # U(img)

    # 1.2 L_0 (LR search image) blurred out version of og LR
    hd, wd= math.ceil(H/s), math.ceil(W/s)
    assert hd > 0 and wd > 0, "Image dimensions cannot be zero!"
    D_I0= resize(I_0, (hd, wd), interp=interp_type, kernel_func= quadratic_bspline)
    L_0 = resize(D_I0, (H, W), interp=interp_type, kernel_func= cubic_bspline)      # U(D(img))
    wpd, hpd= math.ceil(patch_size/s), math.ceil(patch_size/s)  # patch size in LR search image

    assert wpd > 0, "Example Patch size must be greater than 0!"
    assert wpd <= search_size, "Search area must be greater than patch area!"

    # 1.3 HF Detail layer
    H_0= I_0 - L_0

    # --- 2. Create Accumulation Buffers --- #
    print("Step 2: Initializing accumulator buffers...")
    accumulation_buffer = np.zeros_like(L_1)
    weight_buffer = np.zeros(L_1.shape)

    # --- 3. Define Patch Selection mechanism --- #
    print("Step 3: Patch details...")
    for i_h in range(0, hu, stride):
        for j_h in range(0, wu, stride):
            if i_h + patch_size >= hu or j_h + patch_size >= wu:
                continue
            # 3.1 extract HR patch (p)
            p= L_1[i_h: i_h + patch_size, j_h: j_h + patch_size]

            # 3.2 Downscale patch to LR size (we gonna exploit self similarity)
            pd= resize(p, (hpd, wpd), interp='cubic')

            # 3.3 Find relevent search area in L_0 image
            i_c= i_h + patch_size//2
            j_c= j_h + patch_size//2
            search_area_lr= get_search_area(L_0, int(i_c/s), int(j_c/s), wpd)
            search_area_hr= get_search_area(H_0, int(i_c/s), int(j_c/s), wpd)

            # 3.4 Get relevent HF detail
            h_patch= get_best_detail_patch(pd, search_area_lr, search_area_hr)

            # 3.5 Add detail to target
            h_up= resize(h_patch, (patch_size, patch_size), interp= 'cubic')
            p_enhanced= p + h_up

            # 3.6 add to accumulated buffer
            accumulation_buffer[i_h: i_h + patch_size, j_h: j_h + patch_size] += p_enhanced
            weight_buffer[i_h: i_h + patch_size, j_h: j_h + patch_size] += 1
    
    print("Step 4: Finalizing things...")
    weight_buffer[weight_buffer == 0] = 1

    I_1= accumulation_buffer/weight_buffer

    end_time = time.time()
    print(f"Upscaling finished in {end_time - start_time:.2f} seconds.\n")

    return np.clip(I_1, 0., 1.)


