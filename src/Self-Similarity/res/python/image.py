import numpy as np
import cv2
import matplotlib.pyplot as plt
from .constants import *

def load_image(filepath, grayscale= False, normalize=False):
    '''
    **Inputs**
    ----
    - filepath [str]: The filepath to the image
    - grayscale [bool]: Flag to convert the image to grayscale
    - normalize [bool]: converts to 0-1 range

    **Outputs**
    ----
    - img [np.ndarray]: A 2D array, which is the image matrix

    **Description**
    ----
    Reads the image file and gives it as an np.ndarray depending on whether
    you want it as grayscale or rgb
    '''
    ## error checks
    assert filepath is not None, "filepath cannot be \33[1mNone\33[0m!"

    ## read as grayscale if required
    img= cv2.imread(filepath, flags= cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR_RGB)

    ## error checks
    assert img is not None, f"Could not read from \33[1;35m{filepath}\33[0m!"

    return img.astype(dtype=np.int32) if not normalize else img/255.0

def display_image(img, title= 'title', cmap= 'gray'):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix
    - title [str]: Title to the plot of the image
    - cmap [str]: cmap for `plt.imshow`, by default is 'gray'

    **Outputs**
    ----
    None

    **Description**
    ----
    Displays the image stored in the `img` buffer using
    `plt.imshow`.
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
    assert str is not None and isinstance(title, str), "\33[1mtitle\33[0m must be a string"
    

    ## display routine
    plt.imshow(img, cmap= cmap)
    plt.title(title)
    plt.show()
    

def avg_intensity(img):
    '''
    **Inputs**
    ----
    - img [np.ndarray]: Image matrix of size M x N
  

    **Outputs**
    ----    
    - avg [np.float64]: average intensity of pixels (aka image average)

    **Description**
    ----
    Calculates average intensity of the image
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
    
    return np.sum(img)/img.size

def offset_image(img, offset):
    '''
    **Inputs**
    ----
    - img [np.ndarray | List]: Image matrix of size M x N
    - offset [int, np.integer]: offset for the image

    **Outputs**
    ----    
    - img_off [np.ndarray]: image with offset

    **Description**
    ----
    Returns the image with offset, clamping values between 0 and K-1
    '''
    ## error checks
    assert img is not None, "Image matrix cannot be \33[1mNone\33[0m!"
    assert isinstance(img,(np.ndarray,)) and len(img.shape) >= 2, "\33[1mimg\33[0m must be a numpy.ndarray with dimesion more than 1"
    assert np.isdtype(img.dtype, np.integer), "image must be of integral dtype!"
    assert isinstance(offset, (int, np.integer)), "\33[1moffset\33[0m must be integral"
    
    return np.clip(img+offset, 0, K-1)


