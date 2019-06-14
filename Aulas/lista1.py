#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 17:20:24 2019

@author: icaro
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np

#%% Q1. Desenvolva um código capaz de gerar uma imagem 400x400 toda preta e uma
#   imagem 400x400 toda cinza (128, 128, 128), como as ilustradas abaixo.

img = np.zeros((400,400), dtype=np.uint8)
img2 = img + 128
cv2.imshow("preto", img)
cv2.imshow("cinza", img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

#%% Q2.         
img = cv2.imread("baboon.png", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# linhas, colunas
img2 = img[250:450, 125:375]
plt.imshow(img2)
cv2.waitKey(0)
cv2.destroyAllWindows()