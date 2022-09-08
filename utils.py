import cv2
import numpy as np


def remove_small_objects(image, threshold):
    """
    funkcja do usuwania małych obiektów
    args:
        - image - obraz z którego chcemy usunąć obiekty
        - threshold - minimalna wielkość (długość) konturu który zakwalifikuje się do usunięcia
    returns:
        - image - obraz bez zbędnych obiektów
    """

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
    """
    funkcja zwracająca kontury obiektów z obrazu
    args:
        - image - obraz na którym ma zostać przeprowadzona detekcja konturów
    returns:
        - cnts - kontury obiektów
    """
    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnts = cnts[0]
    if cnts:
        return cnts
    else:
        return ()


def check_contour_area(contour):
    coordinates = []
    coordinate = np.reshape(contour, [-1, 2])
    top, left = np.amin(coordinate, axis=0)
    right, bottom = np.amax(coordinate, axis=0)
    area = cv2.contourArea(contour)
    if area > 200:
        coordinates.append([(top - 2, left - 2), (right + 1, bottom + 1)])

    return coordinates
