import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% read a grayscale image
img = cv2.imread('macaco.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img, cmap='gray')

#%% read a color image
img = cv2.imread('macaco.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)

#%% visualizing the tone of each color in image
rgb = cv2.split(img)
plt.subplot(221), plt.imshow(img)
plt.subplot(222), plt.title('R'), plt.imshow(rgb[0], cmap='gray')
plt.subplot(223), plt.title('G'), plt.imshow(rgb[1], cmap='gray')
plt.subplot(224), plt.title('B'), plt.imshow(rgb[2], cmap='gray')
plt.show()

#%% visualizing the histogram of a grayscale image
img = cv2.imread('macaco.jpg', cv2.IMREAD_GRAYSCALE)
plt.subplot(121), plt.title('Original'), plt.imshow(img, cmap='gray')
plt.subplot(122), plt.title('Histogram'), plt.hist(img.ravel(), 256, [0, 256])
plt.show()

#%% visualizing the histogram of a color image
img = cv2.imread('macaco.jpg')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()