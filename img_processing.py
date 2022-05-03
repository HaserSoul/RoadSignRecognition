import cv2
import numpy as np

def expose_blue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([53, 81, 0])
    upper_blue = np.array([160, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_and(img, img, mask=mask_blue)

    return mask

def expose_yellow(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([12,39,1])
    upper_yellow = np.array([25,91,198])

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask = cv2.bitwise_and(img, img, mask=mask_yellow)

    return mask

def expose_orange(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([0,122,61])
    upper_orange = np.array([22,255,255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    mask = cv2.bitwise_and(img, img, mask=mask_orange)

    return mask   


def expose_red(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([125,55,106])
    upper_red = np.array([255,255,255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_and(img, img, mask=mask_red)

    return mask



def LaplacianOfGaussian(image):
    blur = cv2.GaussianBlur(image, (7,7), 1) 
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplacian_edges = cv2.Laplacian(gray, cv2.CV_8U, 3,1,3)
    kernel = np.ones((1,1))
    img_dil = cv2.dilate(laplacian_edges, kernel=kernel, iterations=1)
    return img_dil



def binarization(image):
    thresh = cv2.threshold(image, 15, 255, cv2.THRESH_BINARY)[1]
    return thresh


def preprocess_image(image):
    image = binarization(image)
    image = LaplacianOfGaussian(image)

    return image