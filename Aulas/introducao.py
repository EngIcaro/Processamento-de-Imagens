#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:23:55 2019

@author: icaro
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

#%% Read a grey scale image
img = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)              #cv2.imshow('imagem', img)
plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# Load a color image
img = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Load a color image and visualize each channel separately
img = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
rgb = cv2.split(img)

#subplot(nrows, ncols, index, **kwargs)
plt.subplot(1,4,1); plt.title("Original"); plt.imshow(img)
plt.subplot(1,4,2); plt.title("Red"); plt.imshow(rgb[0], cmap='gray')
plt.subplot(1,4,3); plt.title("Blue"); plt.imshow(rgb[1], cmap='gray')
plt.subplot(1,4,4); plt.title("Green"); plt.imshow(rgb[2], cmap='gray')
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Visualizing histogram of a greyscale image
img = cv2.imread("baboon.png", cv2.IMREAD_GRAYSCALE)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(1,2,1); plt.title("Original"); plt.imshow(img)
plt.subplot(1,2,2); plt.title("Histogram"); plt.hist(img.ravel(), 256, [0,256])
plt.show()

#%%Visualizing histogram of a color image
img = cv2.imread('baboon.png')
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color = col)
    plt.xlim([0, 256])
plt.show()

#%% CREATING IMAGES

# Creating an image filled with zeros
 img = np.zeros((400, 400), dtype=np.float64)
 cv2.imshow("img", img)
 cv2.waitKey(0)
 cv2.destroyAllWindows()

#%% Creating an image filled with ones
img = np.ones((400, 400), dtype=np.float64)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Initializing a grayscale image with random values, uniformly distributed
img = np.ones((400, 400), dtype=np.uint8)
# randu(dst, low, high) → None
# dst – output array of random numbers; the array must be pre-allocated.
# low – inclusive lower boundary of the generated random numbers.
# high – exclusive upper boundary of the generated random numbers.
cv2.randu(img, 0, 255)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#%% Initializing a color image with random values, uniformly distributed
img = np.ones((400, 400, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img = cv2.merge(bgr)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Initializing a grayscale image with random values, normally distributed
img = np.ones((400, 400), dtype=np.uint8)
cv2.randn(img, 127, 40)
cv2.imshow("img", img)
cv2.waitKey(0)
v2.destroyAllWindows()

#%% Initializing a color image with random values, normally distributed
img = np.ones((400, 400, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randn(bgr[0], 127, 40)
cv2.randn(bgr[1], 127, 40)
cv2.randn(bgr[2], 127, 40)
img = cv2.merge(bgr)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%%
# Initialize a color grayscale with uniformly distributed random values and visualize its histogram
img = np.ones((400, 400), dtype=np.uint8)
cv2.randu(img, 0, 255)
histr = cv2.calcHist([img], [], None, [255], [0, 255])
plt.plot(histr)
plt.xlim([0, 256])
plt.show()

#%%

# Initialize a color image with uniformly distributed random values and visualize its histogram
img = np.ones((200, 200, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randu(bgr[0], 0, 255)
cv2.randu(bgr[1], 0, 255)
cv2.randu(bgr[2], 0, 255)
img = cv2.merge(bgr)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [255], [0, 255])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

#%%

# Initialize a grayscale image with normally distributed random values and visualize its histogram
img = np.ones((400, 400), dtype=np.uint8)
cv2.randn(img, 127, 40)
histr = cv2.calcHist([img], [], None, [255], [0, 255])
plt.plot(histr)
plt.xlim([0, 256])
plt.show()

#%%

# Initialize a color image with normally distributed random values and visualize its histogram
img = np.ones((400, 400, 3), dtype=np.uint8)
bgr = cv2.split(img)
cv2.randn(bgr[0], 127, 40)
cv2.randn(bgr[1], 127, 40)
cv2.randn(bgr[2], 127, 40)
img = cv2.merge(bgr)
color = ('b', 'g', 'r')
for i, col in enumerate(color):
    histr = cv2.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(histr, color=col)
    plt.xlim([0, 256])
plt.show()

#%% Convert image to different ranges
img = np.ones((3, 3), dtype=np.float32)

cv2.randn(img, 0.5, 0.5)
print("Normally distributed random values = \n", img, "\n\n")

cv2.normalize(img, img, 255, 0, cv2.NORM_MINMAX)
print("Normalized = \n", img, "\n\n")

img = np.asarray(img, dtype=np.uint8)
print("Converted to uint8 = \n", img, "\n\n")

img = 255 * img
img = np.asarray(img, dtype=np.uint8)
print(img, "\n\n")

#%%

# Create random images continuously
 img = np.ones((250, 250), dtype=np.uint8)
 while cv2.waitKey(1) != ord('q'):
     cv2.randn(img, 120, 60)
     cv2.imshow("img", img)
     
 
 cv2.destroyAllWindows()    


