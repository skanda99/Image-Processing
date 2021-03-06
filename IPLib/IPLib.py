
#   Image Processing Library

import numpy as np
import itertools as it
import cv2
import fractions as fr
import matplotlib.pyplot as plt
import random


# Attribute 1

""" All the kernels used in the Library. """
kernel_bank = {
                'blur_small': np.ones((7,7),dtype=float) * 1/49.0,
                'blur_large': np.ones((21,21),dtype=float) * 1/441.0,
                'blur_weight': np.array([[1,1,1],[1,10,1],[1,1,1]],dtype=float) * 1/18.0,
                'gaussian_3x3': np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=float) * 1/16.0,
                'gaussian_5x5': np.array([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]],dtype=float) * 1/256.0,
                'edge_default': np.array([[1,0,-1],[0,0,0],[-1,0,1]],dtype=float),
                'positive_laplacian': np.array([[0,1,0],[1,-4,1],[0,1,0]],dtype=float),
                'negative_laplacian': np.array([[0,-1,0],[-1,4,-1],[0,-1,0]],dtype=float),
                'laplacian_2': np.array([[1,1,1],[1,-6,1],[1,1,1]],dtype=float) * 1/1.0,
                'prewitt_horizontal': np.array([[-1,-1,-1],[0,0,0],[1,1,1]],dtype=float),
                'prewitt_vertical': np.array([[-1,0,1],[-1,0,1],[-1,0,1]],dtype=float),
                'sobel_horizontal': np.array([[-1,-2,-1],[0,0,0],[1,2,1]],dtype=float),
                'sobel_vertical': np.array([[-1,0,1],[-2,0,2],[-1,0,1]],dtype=float),
                'sharp': np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=float)
                }


# Function 1
def color2gray (img,w_r=0.2989,w_g=0.5870,w_b=0.1140):
    """
        Converts RGB images to gray Scale. Default values of weights are according to the
        standard formula. Returned image has pixel values in the range [0,255].
    """

    gray = w_b*img[:,:,0] + w_g*img[:,:,1] + w_r*img[:,:,2]
    gray = np.round(gray,0)
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
    cv2.imshow(name,img.astype(int)/255.0)
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

    pad_m = m * (stride-1) + r -1
    pad_n = n * (stride-1) + r -1

    img_pad = np.zeros((m+pad_m,n+pad_n),dtype=float)
    img_pad[:m,:n] = np.copy(img)

    if pad_type == 'zero_pad':
        return conv_2D(img_pad,kernel,stride)       # define

    elif pad_type == 'replicate_pad':

        for i in range(m,m+pad_m):
            img_pad[i,:] = np.copy(img_pad[m-1,:])

        for j in range(n,n+pad_n):
            img_pad[:,j] = np.copy(img_pad[:,n-1])

        return conv_2D(img_pad,kernel,stride)       # define

    elif pad_type == 'wrap_pad':

        for i in range(m,m+pad_m):
            img_pad[i,:] = np.copy(img_pad[i-m,:])

        for j in range(n,n+pad_n):
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

    kernel = np.flip(kernel,axis=1)
    kernel = np.flip(kernel,axis=0)

    c_m, c_n = int(np.ceil((m-r+1)/stride)), int(np.ceil((n-c+1)/stride))
    img_conv = np.zeros((c_m,c_n),dtype=float)

    for i,j in it.product(range(c_m),range(c_n)):
        img_conv[i,j] = (img[i*stride:i*stride+r,j*stride:j*stride+c] * kernel).sum()

    return img_conv


# Function 10
def blur_img(img,key='blur_small',stride=1,pad_type='None'):
    """
        Returns blurred version of passed image. Three options for kernels - small, large, weighted.
        Pad_type determines kind of padding to be used to retain original size of the image.
    """

    kernel = kernel_bank[key]
    return img_conv_2D(img,kernel,stride,pad_type)


# Function 11
def sharp_img(img,stride=1,pad_type='None'):
    """
        Returns sharpened version of passed image using laplacian kernel. Pad_types - zero_pad, wrap_pad, replicate_pad
    """

    kernel = kernel_bank['sharp']
    return img_conv_2D(img,kernel,stride,pad_type)


# Function 12
def gaussian_blur(img,key='gaussian_3x3',stride=1,pad_type='None'):
    """
        Returns blurred version of passed image. Gaussian kernel is used for blurring.
        Two gaussian kernels are present - 3x3, 5x5
    """

    kernel = kernel_bank[key]
    return img_conv_2D(img,kernel,stride,pad_type)


# Function 13
def detect_edge(img,key='edge_default',stride=1,pad_type='None'):
    """
        Returns edge detected version of passed image. Kernels available - default,
        laplacian(positive, negative), prewitt (horizontal, vertical), sobel(horizontal,vertical).
    """

    kernel = kernel_bank[key]
    return img_conv_2D(img,kernel,stride,pad_type)


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


# Function 19
def img_contrast(img):
    """
        Returns the contrast of input grayscale image.
    """

    return img.max()-img.min()



# Function 20
def change_brightness(img,k=0):
    """
        Returns an image with changed brightness.
    """

    img_copy = np.copy(img)
    img_copy = img_copy.astype(int)
    img_copy += k

    return img_copy


# Function 20
def img_histogram(img):
    """
        Draws histogram of image pixel values.
        Images can be grayscale or color.
    """

    plt.figure()

    if len(img.shape) > 2:

        plt.subplot(3,1,1)
        plt.hist(img[:,:,0].ravel(),bins=range(257),color='b')
        plt.title('Image Histogram')
        plt.legend('Blue')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')

        plt.subplot(3,1,2)
        plt.hist(img[:,:,1].ravel(),bins=range(257),color='g')
        plt.legend('Green')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')

        plt.subplot(3,1,3)
        plt.hist(img[:,:,2].ravel(),bins=range(257),color='r')
        plt.legend('Red')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')

        plt.ion()
        plt.show()

    else:

        plt.hist(img[:,:].ravel(),bins=range(257))
        plt.title('Image Histogram - Grayscale')
        plt.xlabel('Pixel Values')
        plt.ylabel('Frequency')

        plt.ion()
        plt.show()


# Function 21
def dither_img(img,num_pxl,bw_threshold=128):
    """
        First performs binary thresholing on gray scale image, then dithering and return dithered image.
    """

    img_copy = np.copy(img)
    img_copy[img_copy >= bw_threshold] = 255
    img_copy[img_copy < bw_threshold] = 0

    h = img_copy.shape[0]
    w = img_copy.shape[1]

    coordinates_0 = np.where(img_copy == 0)
    coordinates_0 = tuple(zip(coordinates_0[0],coordinates_0[1]))

    coordinates_255 = np.where(img_copy == 255)
    coordinates_255 = tuple(zip(coordinates_255[0],coordinates_255[1]))

    if num_pxl == 0:
        return img_copy

    selected_coordinates_0 = random.sample(coordinates_0,min(num_pxl,min(len(coordinates_0),len(coordinates_255))))
    selected_coordinates_255 = random.sample(coordinates_255,min(num_pxl,min(len(coordinates_0),len(coordinates_255))))

    selected_coordinates_0 = tuple(zip(*selected_coordinates_0))
    selected_coordinates_255 = tuple(zip(*selected_coordinates_255))

    img_copy[selected_coordinates_0[0],selected_coordinates_0[1]] = 255
    img_copy[selected_coordinates_255[0],selected_coordinates_255[1]] = 0

    return img_copy


# Function 22
def histogram_stretching(img):
    """
        Increases contrast of image by proportionally increasing pixel values.
        Also known as histogram stretching.
    """

    img_copy = np.copy(img)

    img_min = img_copy.min()
    img_max = img_copy.max()

    if img_min == img_max:
        return None

    img_copy = (img_copy-img_min)/(img_max-img_min) * 255

    return img_copy


# Function 23
def histogram_equalize(img):
    """
        Performs histogram equalization, can increase or decrease contrast.
        Returns new image with changed pixel values.
    """

    img_copy = np.copy(img)

    elements,counts = np.unique(img_copy,return_counts=True)
    pdf = counts/counts.sum()
    cdf = np.cumsum(pdf)
    new_values = cdf * 255

    old_new_map = dict(zip(elements,new_values))

    img_new = np.zeros(img_copy.shape)
    for i in old_new_map:
        img_new[img_copy == i] = old_new_map[i]

    return img_new


# Function 24
def histogram_equalization(img):
    """
        Handles histogram equalization for both RGB and gray scale images.
    """

    if len(img.shape) == 3:
        img_copy = np.copy(img)

        blue = img_copy[:,:,0]
        blue = histogram_equalize(blue)

        green = img_copy[:,:,1]
        green = histogram_equalize(green)

        red = img_copy[:,:,2]
        red = histogram_equalize(red)

        new_img = np.zeros(img_copy.shape)

        new_img[:,:,0] = blue
        new_img[:,:,1] = green
        new_img[:,:,2] = red

        return new_img

    else:
        return histogram_equalize(img)


# Function 25
def linear_transform(img):
    """
        Transformation: y = f(x) = x
    """

    img_copy = np.copy(img)
    return img_copy


# Function 26
def negative_transform(img):
    """
        Transformation: y = f(x) = 255-x
    """

    img_copy = np.copy(img)
    img_copy = 255 - img_copy

    return img_copy


# Function 27
def log_transform(img,scale_factor=1):
    """
        Transformation: y = f(x) = scale_factor * log(x + 1)
    """

    img_copy = np.copy(img)
    img_copy = scale_factor * np.log(img_copy+1)

    return img_copy


# Function 28
def inverse_log_transform(img,scale_factor=1):
    """
        Transformation: y = f(x) = scale_factor * exp(x)
    """

    img_copy = np.copy(img)
    img_copy = scale_factor * np.exp(img_copy)

    return img_copy


# Function 29
def gamma_transform(img,scale_factor=1,gamma=2.5):
    """
        Transformation: y = f(x) = scale_factor * x ^ (1/gamma)
        Also known as power transformation.
    """

    img_copy = np.copy(img)
    img_copy = scale_factor * img_copy ** (1/gamma)

    return img_copy


# Function 30
def gaussian_filter(size,sigma=-1):
    """
        returns approximate gaussian filter for given size and sigma with mean 0.
        if sigma is -1, then sigma is automatically set to sqrt(size)
    """

    if sigma == -1:
        sigma = np.sqrt(size)

    filter = np.zeros((size,size))

    for i,j in it.product(range(size),range(size)):
         x = j-size//2
         y = i-size//2
         filter[i,j] = 1/(2*np.pi*sigma**2) * np.exp(-(x**2+y**2)/(2*sigma**2))

    filter = filter/filter[0,0]
    filter = filter/filter.sum()

    return filter


# Function 31
def non_max_suppression_canny(gradient,direction):
    """
        Applies non-max suppression on image gradients to remove repeating pixels.
        Returns new image with suppressed gradients.
    """

    m,n = gradient.shape
    new_img = np.zeros((m,n))
    PI = np.pi

    for i,j in it.product(range(1,m-1),range(1,n-1)):

        if (direction[i,j] < PI/8 and direction[i,j] >= 15*PI/8) or (direction[i,j] < 9*PI/8 and direction[i,j] >= 7*PI/8):
            adj1_mag = gradient[i,j-1]
            adj2_mag = gradient[i,j+1]

        elif (direction[i,j] < 3*PI/8 and direction[i,j] >= PI/8) or (direction[i,j] < 11*PI/8 and direction[i,j] >= 9*PI/8):
            adj1_mag = gradient[i-1,j+1]
            adj2_mag = gradient[i+1,j-1]

        elif (direction[i,j] < 5*PI/8 and direction[i,j] >= 3*PI/8) or (direction[i,j] < 13*PI/8 and direction[i,j] >= 11*PI/8):
            adj1_mag = gradient[i-1,j]
            adj2_mag = gradient[i+1,j]

        else:
            adj1_mag = gradient[i-1,j-1]
            adj2_mag = gradient[i+1,j+1]


        if gradient[i,j] >= adj1_mag and gradient[i,j] >= adj2_mag:
            new_img[i,j] = gradient[i,j]

    return new_img


# Function 32
def dfs_canny(i,j,img,vis,weak,m,n):
    """
        Performs DFS on image to check if weak pixels are connected to strong pixels.
        Part of hysterisis canny. Returns changed image.
    """
    if i >= 0 and i < m and j >= 0 and j < n and vis[i][j] == False and img[i,j] != 0:

        if img[i,j] == weak:
            img[i,j] = 255

        vis[i][j] = True

        dfs_canny(i,j-1,img,vis,weak,m,n)
        dfs_canny(i,j+1,img,vis,weak,m,n)
        dfs_canny(i-1,j,img,vis,weak,m,n)
        dfs_canny(i+1,j,img,vis,weak,m,n)
        dfs_canny(i-1,j-1,img,vis,weak,m,n)
        dfs_canny(i-1,j+1,img,vis,weak,m,n)
        dfs_canny(i+1,j-1,img,vis,weak,m,n)
        dfs_canny(i+1,j+1,img,vis,weak,m,n)


# Function 33
def hysterisis_canny(img,weak):
    """
        Performs hysterisis to decide if weak pixels are part of edges or not.
    """

    m,n = img.shape
    img_copy = np.copy(img)
    vis = [[False for j in range(n)] for i in range(m)]

    for i,j in it.product(range(m),range(n)):
        if img_copy[i,j] == 255:
            dfs_canny(i,j,img_copy,vis,weak,m,n)

    img_copy[img_copy < 255] = 0

    return img_copy


# Function 34
def canny_edge_detection(img,weak=50,high=70,low=30,kernel_size=7,stride_gauss=1,stride_sobel=1,pad_type_gauss='zero_pad',pad_type_sobel='None'):
    """
        Performs canny edge detection. Processes followed:
            1. Reducing noise using gaussian filter
            2. Identify gradient magnitude and direction in image using sobel kernel
            3. Non-Max suppression to identify repeating edges
            4. Thresholding to identify strong and weak pixels
            5. Hysterisis to convert relevant weak pixels to strong

        Returns image with detected edges.
    """

    img_copy = np.copy(img)
    gauss_kernel = gaussian_filter(kernel_size)

    img_copy = img_conv_2D(img_copy,gauss_kernel,stride_gauss,pad_type_gauss)

    img_horizontal = detect_edge(img_copy,'sobel_horizontal',stride_sobel,pad_type_sobel)
    img_vertical = detect_edge(img_copy,'sobel_vertical',stride_sobel,pad_type_sobel)

    img_gradient_mag = np.sqrt(img_horizontal**2 + img_vertical**2)
    img_gradient_mag = img_gradient_mag * 255/img_gradient_mag.max()
    img_gradient_dir = np.arctan2(img_horizontal,img_vertical) + np.pi

    img_new = non_max_suppression_canny(img_gradient_mag,img_gradient_dir)

    img_new[img_new >= high] = 255
    img_new[np.logical_and(img_new < high,img_new >= low)] = weak
    img_new[img_new < low] = 0

    img_new = hysterisis_canny(img_new,weak)

    return img_new


# Function 35
def get_seed_points(img,seed_values):
    """
        returns list of coordinates which have required seed values.
    """

    m,n = img.shape
    coordinates = [(i,j) for i,j in it.product(range(m),range(n)) if img[i,j] in seed_values]

    return coordinates


# Function 36
def sgr_segmentation(img,seed_values,error_threshold=5):
    """
        Performs seeded growing regions segmentation. Supply seed values using
        image histogram. Accordingly adjust error_threshold. Supports only gray-scale images.
    """

    img_copy = np.copy(img)

    m,n = img_copy.shape

    seed_points = get_seed_points(img_copy,seed_values)

    vis = [[False for j in range(n)] for i in range(m)]

    for i,j in seed_points:
        if vis[i][j] == False:
            dfs_sgr_segmentation(i,j,img_copy,vis,error_threshold,m,n)

    for i,j in it.product(range(m),range(n)):
        if img_copy[i,j] not in seed_values:
            img_copy[i,j] = 0

    return img_copy


# Function 37
def dfs_sgr_segmentation(i,j,img,vis,error_threshold,m,n):
    """
        Performs DFS to find connected pixels using condition abs(p1-p2) <= threshold
    """

    if vis[i][j] == False:

        vis[i][j] = True

        if i-1 >= 0 and np.abs(img[i,j]-img[i-1,j]) <= error_threshold and vis[i-1][j] == False:
            img[i-1,j] = img[i,j]
            dfs_sgr_segmentation(i-1,j,img,vis,error_threshold,m,n)

        if i+1 < m and np.abs(img[i,j]-img[i+1,j]) <= error_threshold and vis[i+1][j] == False:
            img[i+1,j] = img[i,j]
            dfs_sgr_segmentation(i+1,j,img,vis,error_threshold,m,n)

        if j-1 >= 0 and np.abs(img[i,j]-img[i,j-1]) <= error_threshold and vis[i][j-1] == False:
            img[i,j-1] = img[i,j]
            dfs_sgr_segmentation(i,j-1,img,vis,error_threshold,m,n)

        if j+1 < n and np.abs(img[i,j]-img[i,j+1]) <= error_threshold and vis[i][j+1] == False:
            img[i,j+1] = img[i,j]
            dfs_sgr_segmentation(i,j+1,img,vis,error_threshold,m,n)
