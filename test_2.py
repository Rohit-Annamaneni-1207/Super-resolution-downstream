from PIL import Image
import numpy as np

A = np.array(Image.open('test_outputs/meerkat_map.png').convert('L'))
B = np.array(Image.open('test_samples/meerkat_bicubic.png').convert('L'))

mse = np.mean((A - B) ** 2)
print(f"MSE between MAP output and Bicubic image: {mse}")