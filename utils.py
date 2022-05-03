import cv2
import numpy as np

def remove_small_objects(image, threshold):
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
        image, connectivity=4
    )
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    image = np.zeros((output.shape), dtype=np.uint8)
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            image[output == i + 1] = 255
    return image


def get_contours(image):
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0]
    if cnts:
        return cnts
    else:
        return ()



