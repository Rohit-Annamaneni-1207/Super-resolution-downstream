# from PIL import Image
from src.iterative_backprojection import IBP
import numpy as np
import cv2

image = cv2.imread('test_samples/meerkat.png', cv2.IMREAD_GRAYSCALE)
image = np.array(image)
factor = 2
bicubic_image = cv2.resize(image, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
cv2.imwrite('test_samples/meerkat_bicubic.png', bicubic_image)
cv2.imwrite('test_samples/meerkat_lr.png', image)
hr_image = IBP(image, factor, factor)
cv2.imwrite('test_outputs/meerkat_ibp.png', hr_image)

