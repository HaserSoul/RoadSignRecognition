import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
import keras
from predictions import get_class_name
from img_processing import *
from cropping import cropContour
from utils import *
from static import *


def check_contour(perimeter, centroid, threshold):
    result = []
    for p in perimeter:
        p = p[0]
        distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
        result.append(distance)
    max_value = max(result)
    signature = [float(dist) / max_value for dist in result]

    temp = sum((1 - s) for s in signature)
    temp = temp / len(signature)
    if temp < threshold: 
        return True, max_value + 2
    else:  
        return False, max_value + 2


def find_sign_coordinates(contours, threshold, distance_theshold):
    coordinates = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #cX, cY = Center of contour
        is_sign, max_distance = check_contour(c, [cX, cY], 1 - threshold)
        if is_sign and max_distance > distance_theshold:
            coordinate = np.reshape(c, [-1, 2])
            top, left = np.amin(coordinate, axis=0)
            right, bottom = np.amax(coordinate, axis=0)
            area = cv2.contourArea(c)
            if area > 200:
                coordinates.append([(top - 2, left - 2), (right + 1, bottom + 1)])
    return coordinates


def bin_for_color(masked_image, min_size_of_items):

    binary_image = preprocess_image(masked_image)   

    binary_image = remove_small_objects(binary_image, min_size_of_items) 

    return binary_image


def get_coords_from_processed_image(image, min_size_components, similitary_contour_with_circle, show_bin=False):

    coordinates = []

    red_binary = bin_for_color(masked_image=expose_red(image), min_size_of_items=min_size_components)

    blue_binary = bin_for_color(masked_image=expose_blue(image), min_size_of_items=min_size_components)

    yellow_binary = bin_for_color(masked_image=expose_yellow(image), min_size_of_items=min_size_components)

    orange_binary = bin_for_color(masked_image=expose_orange(image), min_size_of_items=min_size_components)
 

    if show_bin:
        cv2.imshow("red_bin", red_binary)
        cv2.imshow("blue_bin", blue_binary)
        cv2.imshow("yellow__bin", yellow_binary)
        cv2.imshow("orange_bin", orange_binary)

    contours_red = get_contours(red_binary)
    contours_blue = get_contours(blue_binary)
    contours_yellow =get_contours(yellow_binary)
    contours_orange = get_contours(orange_binary)

    contours = contours_red+contours_blue+contours_yellow+contours_orange

    coordinates = find_sign_coordinates(contours, similitary_contour_with_circle, 10)

    

    return coordinates


def main():
    model = keras.models.load_model("model_trained")

    cap = cv2.VideoCapture("videos\VID_20220502_131344.mp4")
    # cap.set(1, 250)
    count = 0
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, ((960, 540)))

        # print("Frame:{}".format(count))

        coordinates = get_coords_from_processed_image(frame, 400, 0.8, False)
        if coordinates is not None:
            for coordinate in coordinates:
                try:
                    cv2.rectangle(frame, coordinate[0], coordinate[1], (255, 0, 255), 1)
                    cropped_image = frame[
                        coordinate[0][1] : coordinate[1][1],
                        coordinate[0][0] : coordinate[1][0],
                    ]
                    cropped_image = cv2.resize(cropped_image, (32, 32))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_image = cv2.equalizeHist(cropped_image)
                    cv2.imshow("sign", cropped_image)
                    cropped_image = cropped_image / 255
                    cropped_image = cropped_image.reshape(1, 32, 32, 1)
                    predictions = model.predict(cropped_image)
                    classIndex = np.argmax(model.predict(cropped_image), axis=-1)
                    probabilityValue = np.amax(predictions)

                    color = colors[0] if probabilityValue > 0.8 else colors[1]
                    label = str(get_class_name(str(classIndex[0])))
                    percent_value = str(round(probabilityValue * 100, 2)) + "% "
                    label = percent_value + label

                    if probabilityValue > 0.9:
                        # sign bounding box
                        cv2.rectangle(
                            frame, coordinate[0], coordinate[1], (255, 255, 255), 2
                        )

                        # text bounding box
                        (w, h), _ = cv2.getTextSize(label, font, font_size, 1)
                        cv2.rectangle(
                            img=frame,
                            pt1=(coordinate[0][0], coordinate[0][1] - 20),
                            pt2=(coordinate[0][0] + w, coordinate[0][1]),
                            color=color,
                            thickness=-1,
                        )
                        cv2.putText(
                            img=frame,
                            text=label,
                            org=(coordinate[0][0], coordinate[0][1] - 5),
                            fontFace=font,
                            fontScale=font_size,
                            color=(1, 1, 1),
                            thickness=2,
                        )

                except Exception as e:
                    print(e)
                    pass

        cv2.imshow("Result", frame)
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
