import numpy as np
import cv2
import math


def cubic_bspline(x):
    '''
    This is the upsampling filter U.
    '''
    x= math.fabs(x)
    
    p1= (1.5*x - 2.5)*x*x + 1.0
    p2= ((-0.5*x + 2.5)*x - 4.0)*x + 2.0

    # if both the conditions ain't followed, then answer is 0
    return (0 <= x and x <= 1)*p1 + (1 < x and x <= 2)*p2 

def quadratic_bspline(x):
    '''
    This is the downsampling filter D
    '''
    x= math.fabs(x)

    p1= -2*x*x + 0.5
    p2= 0.5*(x-1.5)*(x-1.5)

    return (0 <= x and x <= 0.5)*p1 + (0.5 < x and x <= 1.5)*p2


def resampler_1d(arr_1d:np.ndarray, out_len, kernel_func):
    '''
    This is 1D resampling to arbitrary scales using a kernel function for interpolation
    '''
    assert arr_1d is not None and arr_1d.ndim >= 1, "Provide a non-empty 1D ndarray for 1D resampling"
    assert kernel_func is not None, "Provide a valid kernel for interpolation in 1D"

    inp_len= len(arr_1d)
    scale= out_len/inp_len
    
    nchannels= 1
    if arr_1d.ndim > 1: nchannels= arr_1d.shape[1]

    res= np.zeros((out_len, nchannels), dtype=np.float64)

    # radius is dependent on kernel, simple branchless programming
    radius= (kernel_func==cubic_bspline)*2 + (kernel_func==quadratic_bspline)*1.5

    for i in range(out_len):
        # find the center in input array
        t_in= i/scale

        # get kernel support (nbd of pixels to read)
        j_min= int(math.floor(t_in - radius))
        j_max= int(math.floor(t_in + radius))

        sum, sum_weight= 0, 0
    
        for j in range(j_min, j_max):
            # this is what is gonna be used to determine the weight of pixel at j
            t_kernel= t_in- j

            # weight for current pixel
            weight= (scale < 1.0)*(kernel_func(t_kernel*scale) * scale) + (scale >= 1.0)*(kernel_func(t_kernel))
            
            # index handling (edge cases), circular convolution
            idx= (j < 0)*(-j) + (j >= inp_len)*((inp_len - 1) - (j - inp_len) - 1) + (j >= 0 and j < inp_len)*j

            sum+= weight * arr_1d[max(0, min(inp_len - 1, idx))]
            sum_weight+= weight
        
        if sum_weight > 0:
            res[i]= sum / sum_weight
    return res

def resampler_2d(arr_2d: np.ndarray, out_shape, kernel_func):
    '''
    Same as above but in 2D
    '''
    assert arr_2d is not None and arr_2d.ndim >= 2, "Provide a non-empty 2D ndarray for 2D resampling"
    assert kernel_func is not None, "Provide a valid kernel for interpolation in 2D"
    assert out_shape is not None and isinstance(out_shape, (list, tuple)) and len(out_shape) == 2, "Give a shape for 2D array"
    
    h_in, w_in= arr_2d.shape[:2]
    h_out, w_out= out_shape

    nchannels= 1
    if arr_2d.ndim > 2: nchannels= arr_2d.shape[2]

    img_temp= np.zeros((h_in, w_out, nchannels), dtype=np.float64)
    img_out=  np.zeros((h_out, w_out,nchannels), dtype=np.float64)

    # process rows
    for i in range(h_in):
        img_temp[i, :, :]= resampler_1d(arr_2d[i,:,:], w_out, kernel_func)
    
    # process cols
    for j in range(w_out):
        img_out[:, j, :]= resampler_1d(img_temp[:,j,:], h_out, kernel_func)
    
    return np.squeeze(img_out, axis=2) if nchannels == 1 else img_out


interpolation_dict= {
    'cubic':    cv2.INTER_CUBIC,
    'bilinear': cv2.INTER_LINEAR,
    'nearest':  cv2.INTER_NEAREST,
    'wavelet': resampler_2d
}

def resize(img: np.ndarray, shape, interp= 'cubic', kernel_func=None):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: LR image (can be grayscale or color both)
    - shape [float]: new dimesions of image in order (height, width)
    - interp [str]: interpolation type

    **Output**:
    ----
    - iout [float]: resized image

    **Desciption**
    ----
    Gives the scaled image
    '''
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
    assert isinstance(shape, (tuple, list)) and len(shape) == 2, "There are only two image dimensions!"

    assert shape[0] > 0 and shape[1] > 0, "Image dimensions after scaling must be greater than 0!"
    interp_type= interpolation_dict[interp]
    
    if interp_type== resampler_2d:
        assert kernel_func is not None, "Provide a valid kernel function for this interpolation method"
        return np.clip(resampler_2d(img, shape, kernel_func), 0.0, 1.0)

    return np.clip(cv2.resize(img, dsize=(shape[1], shape[0]), interpolation=interp_type), 0., 1.)