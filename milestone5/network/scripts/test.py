
from detector import Detector
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# from network.scripts.detector import Detector
from detector import Detector


# detc = Detector("model\model.best.pth")
detc = Detector("model\yolov8_model.pt")
img = np.array(Image.open('image_0.png'))
# img = np.array(Image.open("dataset\est_7.png"))
# img = mpimg.imread('dataset\images\image_0.png')
detector_output, network_vis = detc.detect_single_image(img)
print(detector_output)

print(network_vis)

# print(img.shape)
# imgplot = plt.imshow(img)
# plt.show()

# print(img.shape)
# imgplot = plt.imshow(network_vis)
# plt.show()