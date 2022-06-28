import cv2
import numpy as np

from pre_processing import pre_processing_image

def get_lines(bw_image, lines_thres):
    #making horizontal projections
    hor_proj = cv2.reduce(bw_image, 1, cv2.REDUCE_AVG)

    #making hist - same dimension as horizontal projection - if all = 0 (space), then True, else False
    th = 0 #black pixels threshold values. this represent the space lines
    hist = hor_proj <= th

    #get mean coordinate of white pixels groups
    y_coords = []
    y = 0
    count = 0
    is_space = False

    # print(bw_image.shape[0], lines_thres)
    for i in range(0, bw_image.shape[0]):
        if (not is_space):
            if (hist[i]): #if space is detected, get the first starting y_coordinates and start count at 1
                is_space = True
                count = 1
                y = i
        else:
            if not hist[i]:
                is_space = False
                '''
                when smoothing, thin letters will breakdown, creating a new blank lines or pixels rows, 
                but the count will small so we set a threshold.
                '''
                if (count >= lines_thres):
                    y_coords.append(y / count)
            else:
                y += i
                count += 1
    if count > 0:
        y_coords.append(y / count)
    # print(y_coords)
    return y_coords

def get_lines_median(bw_image):
    #making horizontal projections
    hor_proj = cv2.reduce(bw_image, 1, cv2.REDUCE_AVG)

    th = 0
    hist = hor_proj <= th

    # y_coords = []
    # y = 0
    count = 0
    is_space = False

    #array of counts of each blank rows of each line found
    median_count = []

    #start count median lines in image
    for i in range(0, bw_image.shape[0]):
        if not is_space:
            if hist[i]: #if space is detected, get the first starting y-coordinates and start count at 1 
                is_space = True
                count = 1
        else:
            if not hist[i]:
                is_space = False
                median_count.append(count)
            else:
                count += 1

    median_count.append(count)

    return median_count

def get_lines_thredshold(percent, img_for_det):
    lines_median = get_lines_median(img_for_det)
    lines_median = sorted(lines_median)
    lines_thres = lines_median[int(len(lines_median) / 3)] * (percent / 100.0)
    lines_thres = int(lines_thres)

    return lines_thres

def get_lines_segmentation(img_url):
    img_for_det, img_for_ext = pre_processing_image(img_url)

    #get line threshold to determine how much gap should be considered as the line gap for segmentation line
    lines_thres = get_lines_thredshold(40, img_for_det)
    # print('line thres: ', lines_thres)
    #get coordinates of the lines // segmatation line
    y_coords = get_lines(img_for_det, lines_thres)

    # print("y_coords: ", y_coords)

    #save image with segmentation lines
    img_with_lines = img_for_ext.copy()
    for i in y_coords:
        i = int(i)
        cv2.line(img_with_lines, (0, i), (img_with_lines.shape[1], i), 0, 1)
    cv2.imwrite(img_url[:-4] + '/' + 'img_with_lines.png', img_with_lines)

    return y_coords, img_for_det, img_for_ext