# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 13:35:32 2018

@author: hasee
"""

import SimpleITK as sitk
import numpy as np
'''
This funciton reads a '.mhd' file using SimpleITK and return the image array, origin and spacing of the image.
'''

def load_itk(filename):
    # Reads the image using SimpleITK
    itkimage = sitk.ReadImage(filename)

    # Convert the image to a  numpy array first and then shuffle the dimensions to get axis in the order z,y,x
    ct_scan = sitk.GetArrayFromImage(itkimage)

    # Read the origin of the ct_scan, will be used to convert the coordinates from world to voxel and vice versa.
    origin = np.array(list(reversed(itkimage.GetOrigin())))

    # Read the spacing along each dimension
    spacing = np.array(list(reversed(itkimage.GetSpacing())))

    return ct_scan, origin, spacing

if __name__ == '__main__':
    print(sitk.Version())
#    ct_scan, origin, spacing = load_itk('data/vorts1.mhd')
    
    import skimage.io as io
    img = io.imread('data/vorts1.mhd', plugin='simpleitk')
#    print(img)
    io.imshow(img[0])
#    io.imsave('~skimage_vorts1.mhd', img, plugin='simpleitk')
    io.imsave('~skimage_vorts1.tif', img, plugin='tifffile')
    sitk.WriteImage(sitk.GetImageFromArray(img), "~sitk_vorts1.mhd")