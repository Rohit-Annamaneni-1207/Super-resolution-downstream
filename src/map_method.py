import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.fftpack import dct, idct
import cv2

def cg(A_func, b, x0=None, tol=1e-6, max_iter=200):
    """
    Solve A x = b using Conjugate Gradient method.
    """
    n = len(b)
    if x0 is None:
        x = np.zeros(n)
    else:
        x = x0.copy()

    r = b - A_func(x)
    p = r.copy()
    rs_old = np.sum(r * r)

    iter_conv = max_iter
    for i in range(max_iter):
        # print(f"iter {i+1}")
        Ap = A_func(p)
        alpha = rs_old / (np.sum(p*Ap) + 10**(-4))
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = np.sum(r * r)

        if np.sqrt(rs_new) < tol:
            print(f"CG converged in {i+1} iterations")
            iter_conv = i+1
            break

        p = r + (rs_new / (rs_old + 1e-12)) * p
        rs_old = rs_new

    return x

def dct2(x):
    return dct(dct(x.T, norm='ortho').T, norm='ortho')

def idct2(x):
    return idct(idct(x.T, norm='ortho').T, norm='ortho')

def map_method(lr_img:np.array, factor:int,  tol:float=1e-4, p=0.3, eps=1e-6, lam=1e-3, max_iter:int=50):

    #image to be restored, assume the 1/factor indicates amount of pixels lost in each dimension
    hr_guess = cv2.resize(lr_img, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC).astype(np.float32)

    mask = np.random.rand(hr_guess.shape[0], hr_guess.shape[1]) < (1/factor)

    x = hr_guess.copy()
    b = mask * hr_guess

    for i in range(max_iter):
        y = dct2(x)

        w = p*(eps + y**2)**(p-1)

        def A_func(z):
            term_1 = mask*z
            term_2 = lam*idct2(w* dct2(z))

            return term_1 + term_2
        
        x_next = cg(A_func, b, x0=x)

        diff = np.sum((x_next[0] - x)**2) / np.sum(x**2)
        print(f"MAP Iteration {i+1}, diff: {diff}")
        if diff < tol:
            return x_next

        x = x_next

    return x_next  


img = Image.open('test_samples/meerkat.png').convert('L')
img = np.array(img)

factor = 2
map_image = map_method(img, factor)
plt.imsave('test_outputs/meerkat_map.png', map_image, cmap='gray')



    





    


