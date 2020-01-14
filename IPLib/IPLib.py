
#   Image Processing Library

import numpy as np
import itertools as it
import cv2
import fractions as fr


# Attribute 1

""" All the kernels used in the Library. """
kernel_bank = {
                'blur_small': np.ones((7,7),dtype=float) * 1/49.0,
                'blur_large': np.ones((21,21),dtype=float) * 1/441.0,
                'laplacian_1': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=float),
                'gaussian_3x3': np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=float) * 1/16.0,
                'gaussian_5x5': np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],dtype=float) * 1/256.0,
                'edge_small': np.array([[1,0,-1],[0,0,0],[-1,0,1]],dtype=float),
                'edge_medium': np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=float),
                'edge_large': np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=float),
                'laplacian_2': np.array([[1,1,1],[1,-6,1],[1,1,1]],dtype=float) * 1/1.0
                }


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
# yet to add support for stride
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
        Returns blurred version of passed image. Two options for kernels - small, large.
        Pad_type determines kind of padding to be used to retain original size of the image.
    """

    if key == 'small':
        kernel = np.ones((7,7),dtype=float) * 1/49.0
        return img_conv_2D(img,kernel,1,pad_type)

    elif key == 'large':
        kernel = np.ones((21,21),dtype=float) * 1/441.0
        return img_conv_2D(img,kernel,1,pad_type)


# Function 11
def sharp_img(img,pad_type='None'):
    """
        Returns sharpened version of passed image. Pad_types - zero_pad, wrap_pad, replicate_pad
    """

    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=float)
    return img_conv_2D(img,kernel,1,pad_type)


# Function 12
def gaussian_blur(img,key='3x3',pad_type='None'):
    """
        Returns blurred version of passed image. Gaussian kernel is used for blurring.
        Two gaussian kernels are present - 3x3, 5x5
    """

    if key == '3x3':

        kernel = np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=float) * 1/16.0
        return img_conv_2D(img,kernel,1,pad_type)

    elif key == '5x5':

        kernel = np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],dtype=float) * 1/256.0
        return img_conv_2D(img,kernel,1,pad_type)


# Function 13
def detect_edge(img,key='small',pad_type='None'):
    """
        Returns edge detected version of passed image. Three kernels are available - small,
        medium, large.
    """

    if key == 'small':

        kernel = np.array([[1,0,-1],[0,0,0],[-1,0,1]],dtype=float)
        return img_conv_2D(img,kernel,1,pad_type)

    elif key == 'medium':

        kernel = np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=float)
        return img_conv_2D(img,kernel,1,pad_type)

    elif key == 'large':

        kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],dtype=float)
        return img_conv_2D(img,kernel,1,pad_type)


# Function 14
def threshold_segment(img):
    """
        Returns a segmented image. Based on thresholding technique.
        Thresholds can be changed for better segmentation.
    """

    m,n = img.shape
    g = img.sum()/(m*n)

    segmented = np.zeros((m,n),dtype=float)

    for i,j in it.product(range(m),range(n)):

        if img[i,j] - g >= 100:
            segmented[i,j] = 0
        elif img[i,j] - g >= 50:
            segmented[i,j] = 50
        elif img[i,j] - g >= 0:
            segmented[i,j] = 100
        elif img[i,j] - g >= -50:
            segmented[i,j] = 150
        elif img[i,j] - g >= -100:
            segmented[i,j] = 200
        else:
            segmented[i,j] = 255

    return segmented


# Function 15
def zoom_pxl_replication(img,z_f=1):
    """
        Returns a new zoomed image with zoom factor = z_f.
        Input image can be colored or grayscale.
    """

    try:
        m,n = img.shape

        new_img = np.zeros((z_f*m,z_f*n),dtype=int)

        for i,j in it.product(range(m),range(n)):
            new_img[z_f*i:z_f*i+z_f,z_f*j:z_f*j+z_f] = np.ones((z_f,z_f),dtype=int) * img[i,j]


    except:
        m,n,_ = img.shape

        new_img = np.zeros((z_f*m,z_f*n,3),dtype=int)

        for i,j in it.product(range(m),range(n)):
            new_img[z_f*i:z_f*i+z_f,z_f*j:z_f*j+z_f,0] = np.ones((z_f,z_f),dtype=int) * img[i,j,0]
            new_img[z_f*i:z_f*i+z_f,z_f*j:z_f*j+z_f,1] = np.ones((z_f,z_f),dtype=int) * img[i,j,1]
            new_img[z_f*i:z_f*i+z_f,z_f*j:z_f*j+z_f,2] = np.ones((z_f,z_f),dtype=int) * img[i,j,2]

    return new_img


# Function 16
def zoom_zero_order(img):
    """
        Returns a zoomed image using zero order method.
        Input image can be colored or grayscale.
    """

    try:
        m,n = img.shape

        tmp1 = np.zeros((m,2*n-1),dtype=float)

        for i in range(n):
            tmp1[:,2*i] = np.copy(img[:,i])

        for i in range(1,2*n-1,2):
            tmp1[:,i] = (tmp1[:,i-1]+tmp1[:,i+1])/2

        tmp2 = np.zeros((2*m-1,2*n-1),dtype=float)

        for j in range(m):
            tmp2[2*j,:] = np.copy(tmp1[j,:])

        for j in range(1,2*m-1,2):
            tmp2[j,:] = (tmp2[j-1,:]+tmp2[j+1,:])/2

    except:
        m,n,_ = img.shape

        tmp1 = np.zeros((m,2*n-1,3),dtype=float)

        for i in range(n):
            tmp1[:,2*i,:] = np.copy(img[:,i,:])

        for i in range(1,2*n-1,2):
            tmp1[:,i,:] = (tmp1[:,i-1,:]+tmp1[:,i+1,:])/2

        tmp2 = np.zeros((2*m-1,2*n-1,3),dtype=float)

        for j in range(m):
            tmp2[2*j,:,:] = np.copy(tmp1[j,:,:])

        for j in range(1,2*m-1,2):
            tmp2[j,:,:] = (tmp2[j-1,:,:]+tmp2[j+1,:,:])/2


    return tmp2


# Function 17
def zoom_k(img,k=1):
    """
        Returns k-times zoomed image.
        Input image can be colored or grayscale.
    """

    try:

        m,n = img.shape

        tmp1 = np.zeros((m,n*k-k+1),dtype=float)

        for i in range(n-1):
            tmp1[:,k*i:k*i+k+1] = np.linspace(img[:,i],img[:,i+1],k+1,axis=1)

        tmp2 = np.zeros((m*k-k+1,n*k-k+1),dtype=float)

        for j in range(m-1):
            tmp2[k*j:k*j+k+1,:] = np.linspace(tmp1[j,:],tmp1[j+1,:],k+1,axis=0)

    except:

        m,n,_ = img.shape

        tmp1 = np.zeros((m,n*k-k+1,3),dtype=float)

        for i in range(n-1):
            tmp1[:,k*i:k*i+k+1,:] = np.linspace(img[:,i,:],img[:,i+1,:],k+1,axis=1)

        tmp2 = np.zeros((m*k-k+1,n*k-k+1,3),dtype=float)

        for j in range(m-1):
            tmp2[k*j:k*j+k+1,:,:] = np.linspace(tmp1[j,:,:],tmp1[j+1,:,:],k+1,axis=0)


    return tmp2


# Function 18
def aspect_ratio(img):
    """
        Returns aspect ratio in string format for grayscale images.
    """

    m,n = img.shape
    a_r = fr.Fraction(n,m)

    return str(a_r)
