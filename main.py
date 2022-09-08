import time
import cv2
import numpy as np
from math import sqrt
import keras
from predictions import get_class_name, red_blue_signs_nrs, yellow_orange_signs_nrs
from img_processing import *
from utils import *
from static import *


def check_contour(contour, centroid, threshold):
    """
    funkcja walidująca kontur. Sprawdza jak bardzo zaokrąglony jest obiekt.
    args:
        contour - tablica z wszystkimi współrzędnymi konturu

        centroid - wyliczony środek konturu

        threshold - jak bardzo zaokrąglony musi być obiekt żeby przeszedł walidacje

    returns:
        True/False - w zależności czy obiekt przeszedł walidacje

        max_value + 2 - odległość pomiędzy dwoma najdalszymi pikselami określającego kontur,
            czyli przekątna konturu + 2
    """
    result = []
    for perimeter in contour:
        p = perimeter[0]
        distance = sqrt((p[0] - centroid[0]) ** 2 + (p[1] - centroid[1]) ** 2)
        result.append(distance)

    max_value = max(result)
    signature = [float(distance) / max_value for distance in result]

    roudness = sum(1 - s for s in signature)
    roudness = roudness / len(signature)

    if roudness < threshold: 
        return True, max_value + 2
    else:  
        return False, max_value + 2


def find_sign_coordinates(contours, threshold, distance_threshold):
    """
    args:
        contours - kontur obiektu
        threshold - minimalny stopień zaokrąglenia obiektu
        distance_threshold - minimalna wielkość obiektu
    returns:
        coordinates - koordynaty obiektów, które przeszły walidacje
    """
    coordinates = []
    for c in contours:
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        #cX, cY = Center of contour
        is_sign, max_distance = check_contour(c, [cX, cY], 1 - threshold)
        if is_sign and max_distance > distance_threshold:
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


def get_coords_from_processed_image(image, min_size_components, show_bin=True):

    coordinates = []

    red_binary = bin_for_color(masked_image=expose_red(image), min_size_of_items=min_size_components)

    blue_binary = bin_for_color(masked_image=expose_blue(image), min_size_of_items=min_size_components)

    yellow_binary = bin_for_color(masked_image=expose_yellow(image), min_size_of_items=min_size_components)

    orange_binary = bin_for_color(masked_image=expose_orange(image), min_size_of_items=min_size_components)

    white_binary = bin_for_color(masked_image=expose_white(image), min_size_of_items=min_size_components)

    yellow_and_orange_binary = bin_for_color(masked_image=expose_yellow_and_orange(image, expose_yellow(image), expose_orange(image)), min_size_of_items=min_size_components)

    
    ### circular and square blue/red signs should be caught
    contours_red = get_contours(red_binary)
    contours_blue = get_contours(blue_binary)
    contours_red_blue = contours_red+contours_blue


    red_blue_signs = find_sign_coordinates(contours=contours_red_blue, threshold=0.75, distance_threshold=10)

    for sign_coords in red_blue_signs:
        coordinates.append((sign_coords, "red_blue"))

    ### triangle signs should be caught    
    contours_yellow_orange = get_contours(yellow_and_orange_binary)

    yellow_orange_signs = find_sign_coordinates(contours=contours_yellow_orange, threshold=0.5, distance_threshold=10)

    for sign_coords in yellow_orange_signs:
        coordinates.append((sign_coords, "yellow_orange"))

    contours_white = get_contours(white_binary)

    priorty_signs = find_sign_coordinates(contours=contours_white, threshold=0.75, distance_threshold=10)

    for sign_coords in priorty_signs:
        coordinates.append((sign_coords, "yellow_orange"))

    # cv2.imshow("white_bin", white_binary)
    if show_bin:
        cv2.imshow("red_bin", red_binary)
        cv2.imshow("blue_bin", blue_binary)
        # cv2.imshow("yellow__bin", yellow_binary)
        # cv2.imshow("orange_bin", orange_binary)


    return coordinates

def main():
    model = keras.models.load_model("model_trained_2")

    cap = cv2.VideoCapture("videos/test9.mp4")
    out = cv2.VideoWriter('output10.mp4', -1, 20.0, (960,540))
    # cap.set(1, 250)
    count = 0
    while True:
        success, frame = cap.read()
        frame = cv2.resize(frame, ((960, 540)))

        coordinates_with_sign_types = get_coords_from_processed_image(frame, 400, True)
        if coordinates_with_sign_types:
            for coordinate, sign_type in coordinates_with_sign_types:
                try:
                    cropped_image = frame[
                        coordinate[0][1] : coordinate[1][1],
                        coordinate[0][0] : coordinate[1][0],
                        ]
                    cropped_image = cv2.resize(cropped_image, (32, 32))
                    cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    cropped_image = cv2.equalizeHist(cropped_image)
                    cropped_image = cropped_image / 255
                    cropped_image = cropped_image.reshape(1, 32, 32, 1)
                    predictions = model.predict(cropped_image)
                    class_index = np.argmax(model.predict(cropped_image), axis=-1)
                    probability_value = np.amax(predictions)

                    color = colors[0] if probability_value > 0.8 else colors[1]
                    label = str(get_class_name(str(class_index[0])))
                    percent_value = str(round(probability_value * 100, 2)) + "% "
                    label = percent_value + label

                    if sign_type == "yellow_orange":
                        if probability_value > 0.9 and str(class_index[0]) in yellow_orange_signs_nrs:
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
                        else:
                            continue
                            cv2.rectangle(frame, coordinate[0], coordinate[1], (0, 0, 255), 2)
                            cv2.putText(
                                img=frame,
                                text=label,
                                org=(coordinate[0][0], coordinate[0][1] - 5),
                                fontFace=font,
                                fontScale=font_size,
                                color=(1, 1, 1),
                                thickness=2,
                            )
                    else:
                        if probability_value > 0.9 and str(class_index[0]) in red_blue_signs_nrs:
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
                        else:
                            continue
                            cv2.rectangle(frame, coordinate[0], coordinate[1], (0, 0, 255), 2)
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
                    pass

        cv2.imshow("Result", frame)
        if success:
            out.write(frame)
        count = count + 1
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

if __name__ == "__main__":
    main()
