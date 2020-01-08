
#   Image Processing Library

import numpy as np
import itertools as it
import cv2

# Function 1
def color2gray (img,w_r=0.2989,w_g=0.5870,w_b=0.1140):
    """
        Converts RGB images to gray Scale. Default values of weights are according to the
        standard formula. Returned image has pixel values in the range [0,255].
    """

    gray = np.empty((img.shape[0],img.shape[1]),dtype=float)

    for i,j in it.product(range(gray.shape[0]),range(gray.shape[1])):
        gray[i,j] = (w_b*img[i,j][0] + w_g*img[i,j][1] + w_r*img[i,j][2])

    return gray


# Function 2
def red_channel(img):
    """
        Returns the red channel of the colored image.
        Returned image has pixel values in the range [0,255].
    """

    red = np.zeros(img.shape,dtype=float)

    red[:,:,2] = np.copy(img[:,:,2])

    return red


# Function 3
def green_channel(img):
    """
        Returns the green channel of the colored image.
        Returned image has pixel values in the range [0,255].
    """

    green = np.zeros(img.shape,dtype=float)

    green[:,:,1] = np.copy(img[:,:,1])

    return green


# Function 4
def blue_channel(img):
    """
        Return the blue channel of the colored image.
        Returned image has pixel values in the range [0,255].
    """

    blue = np.zeros(img.shape,dtype=float)

    blue[:,:,0] = np.copy(img[:,:,0])

    return blue


# Function 5
def img_read(name):
    """
        Returns image loaded, ensure that image file is in the same directory as this file.
        Returned image has integer pixel values in the range [0,255]
    """

    img = cv2.imread(name)

    return img


# Function 6
def img_disp(name,img):
    """
        Displays passed image in a window with the title as passed.
        Image should have pixel values within [0,255]
    """
    cv2.imshow(name,img/255.0)
    cv2.waitKey()


# Function 7
def img_save(name,img):
    """
        Saves the passed image into the PWD with the name passed.
        Image should have pixel values within [0,255]
    """
    cv2.imwrite(name,img)


# Function 8
def img_conv_2D(img,kernel,stride=1,pad_type='None'):
    """
        Returns image obtained by 2D-Convolution between img and kernel with stride(defaulted to 1).
        To get the original size as the image, provisions for zero padding, replicate padding and
        wrap padding are given. Both img and kernel are 2D img arrays. The default value of pad_type
        is None.
    """

    m,n = img.shape
    r,c = kernel.shape

    img_pad = np.zeros((m+r-1,n+c-1),dtype=float)
    img_pad[:m,:n] = np.copy(img)

    if pad_type == 'zero_pad':
        return conv_2D(img_pad,kernel,stride)       # define

    elif pad_type == 'replicate_pad':

        for i in range(m,m+r-1):
            img_pad[i,:] = np.copy(img_pad[m-1,:])

        for j in range(n,n+c-1):
            img_pad[:,j] = np.copy(img_pad[:,n-1])

        return conv_2D(img_pad,kernel,stride)       # define

    elif pad_type == 'wrap_pad':

        for i in range(m,m+r-1):
            img_pad[i,:] = np.copy(img_pad[i-m,:])

        for j in range(n,n+c-1):
            img_pad[:,j] = np.copy(img_pad[:,j-n])

        return conv_2D(img_pad,kernel,stride)       # define

    else:

        return conv_2D(np.copy(img),kernel,stride)           # define


# Function 9
def conv_2D(img,kernel,stride=1):
    """
        Performs convolution between img and kernel with given stride (default value 1).
        Returns the output array of convolution operation.
    """

    m,n = img.shape
    r,c = kernel.shape

    img_conv = np.zeros((m-r+1,n-c+1),dtype=float)

    for i,j in it.product(range(m-r+1),range(n-c+1)):
        img_conv[i,j] = (img[i:i+r,j:j+c] * kernel).sum()

    return img_conv


# Function 10
def blur_img(img,key='small',pad_type='None'):
    """
        Blurs the passed image. Two options for kernels. Pad_type determines kind of padding
        to be used to retain original size of the image.
    """

    if key == 'small':
        kernel = np.ones((7,7),dtype=float) * 1/49.0
        return img_conv_2D(img,kernel,1,pad_type)

    elif key == 'large':
        kernel = np.ones((21,21),dtype=float) * 1/441.0
        return img_conv_2D(img,kernel,1,pad_type)
