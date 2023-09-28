
from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


detc = Detector("model\model.best.pth")
img = np.array(Image.open('dataset\images\image_104p.png'))
# img = np.array(Image.open("dataset\est_7.png"))
# img = mpimg.imread('dataset\images\image_0.png')
detector_output, network_vis = detc.detect_single_image(img)


# print(img.shape)
# imgplot = plt.imshow(img)
# plt.show()

print(img.shape)
imgplot = plt.imshow(network_vis)
plt.show()