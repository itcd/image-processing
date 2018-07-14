# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 08:44:32 2018

@author: hasee
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt

WIDTH = 128   # has a great influence on the result

def spectral_residual_saliency_map(img):
    img = cv2.resize(img, (WIDTH,int(WIDTH*img.shape[0]/img.shape[1])))
#    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
#    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
#    plt.title(img.shape)
#    plt.show()
    
    c = cv2.dft(np.float32(img), flags = cv2.DFT_COMPLEX_OUTPUT)
    mag = np.sqrt(c[:,:,0]**2 + c[:,:,1]**2)
    spectralResidual = np.exp(np.log(mag) - cv2.boxFilter(np.log(mag), -1, (3,3)))

    c[:,:,0] = c[:,:,0] * spectralResidual / mag
    c[:,:,1] = c[:,:,1] * spectralResidual / mag
    c = cv2.dft(c, flags = (cv2.DFT_INVERSE | cv2.DFT_SCALE))
    mag = c[:,:,0]**2 + c[:,:,1]**2
    cv2.normalize(cv2.GaussianBlur(mag,(9,9),3,3), mag, 0., 1., cv2.NORM_MINMAX)
    
#    cv2.imshow('Saliency Map', mag)
#    c = cv2.waitKey(0) & 0xFF
#    if(c==27 or c==ord('q')):
#        cv2.destroyAllWindows()

    plt.imshow(mag, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(mag.shape)
    plt.show()
    plt.imsave('~saliency_dancing.jpg', mag, cmap = 'gray')

if __name__ == '__main__':

    img = cv2.imread('images/dancing.jpg', cv2.IMREAD_GRAYSCALE)
    #cv2.imshow('image',img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.title(img.shape)
    plt.show()
    
    spectral_residual_saliency_map(img)
