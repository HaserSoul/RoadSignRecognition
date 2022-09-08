import cv2
import numpy as np

def expose_blue(img):
    """
    funkcja wyciągająca obiekty w odcieniach niebieskiego z obrazu
    args:
        img - klatka wejściowa
    returns:
        mask - klatka z wyciągniętymi obiektami
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([53, 81, 0])
    upper_blue = np.array([160, 255, 255])

    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask = cv2.bitwise_and(img, img, mask=mask_blue)

    return mask

def expose_orange(img):
    """
    funkcja wyciągająca obiekty w odcieniach koloru pomarańczowego z obrazu
    args:
        img - klatka wejściowa
    returns:
        mask - klatka z wyciągniętymi obiektami
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_orange = np.array([0,122,61])
    upper_orange = np.array([22,255,255])

    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    mask = cv2.bitwise_and(img, img, mask=mask_orange)

    return mask   


def expose_red(img):
    """
    funkcja wyciągająca obiekty w odcieniach koloru czerwonego z obrazu
    args:
        img - klatka wejściowa
    returns:
        mask - klatka z wyciągniętymi obiektami
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([125,55,106])
    upper_red = np.array([179,255,255])

    mask_red = cv2.inRange(hsv, lower_red, upper_red)

    mask = cv2.bitwise_and(img, img, mask=mask_red)

    return mask

def expose_yellow(img):
    """
    funkcja wyciągająca obiekty w odcieniach koloru żółtego z obrazu
    args:
        img - klatka wejściowa
    returns:
        mask - klatka z wyciągniętymi obiektami
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([0,53,156])
    upper_yellow = np.array([24,255,255])


    mask = cv2.inRange(hsv,lower_yellow, upper_yellow)

    mask = cv2.bitwise_and(img, img, mask=mask)

    return mask 

def expose_yellow_and_orange(img, yellow_mask, orange_mask):

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20,92,165])
    upper_yellow = np.array([29,255,255])

    more_yellow_mask = cv2.inRange(hsv,lower_yellow, upper_yellow)

    mask = cv2.bitwise_and(yellow_mask, orange_mask) 

    mask = cv2.bitwise_and(img, mask, more_yellow_mask)

    return mask

def expose_white(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([91,0,209])
    upper_white = np.array([179,35,255])


    mask = cv2.inRange(hsv,lower_white, upper_white)

    lower_yellow = np.array([23,112,172])
    upper_yellow = np.array([37,255,235])

    yellow_mask = cv2.inRange(hsv,lower_yellow, upper_yellow)

    mask = cv2.bitwise_or(img, img, mask=mask)

    mask = cv2.bitwise_or(img, mask, mask=yellow_mask)

    more_yellow_lower = np.array([13,52,129])
    more_yellow_upper =  np.array([19,255,255])

    more_yellow_mask = cv2.inRange(hsv,more_yellow_lower, more_yellow_upper)

    mask = cv2.bitwise_or(img, mask, mask=more_yellow_mask)

    return mask  

def preprocess_binary_img(image):
    blur = cv2.GaussianBlur(image, (7,7), 1) 
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    laplacian_edges = cv2.Laplacian(gray, cv2.CV_8U, 3,1,3)
    img_canny = cv2.Canny(gray, 23, 22)
    kernel = np.ones((3,3))
    img_dil = cv2.dilate(img_canny, kernel=kernel, iterations=1)
    return img_dil


def binarization(image):
    thresh = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)[1]
    return thresh


def preprocess_image(image):
    image = binarization(image)
    image = preprocess_binary_img(image)

    return image