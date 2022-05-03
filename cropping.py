# crop sign
def cropContour(image, center, max_distance):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(center[0] - max_distance), 0])
    bottom = min([int(center[0] + max_distance + 1), height - 1])
    left = max([int(center[1] - max_distance), 0])
    right = min([int(center[1] + max_distance + 1), width - 1])
    return image[left:right, top:bottom]


def cropSign(image, coordinate):
    width = image.shape[1]
    height = image.shape[0]
    top = max([int(coordinate[0][1]), 0])
    bottom = min([int(coordinate[1][1]), height - 1])
    left = max([int(coordinate[0][0]), 0])
    right = min([int(coordinate[1][0]), width - 1])
    return image[top:bottom, left:right]
