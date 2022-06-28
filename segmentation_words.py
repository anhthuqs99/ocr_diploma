# from fcntl import F_SEAL_SEAL
import cv2
import numpy as np
import os

from segmentation_lines import get_lines_segmentation

SEG_IMG_LOCATION = './segmentation_images/'

def get_spaces(line, thres_space):
    #making vertical projections
    ver_proj = cv2.reduce(line, 0, cv2.REDUCE_AVG)

    #make hist - same dimension as hor_pro, if it has value 0 (aka space) then hist is True else False
    th = 0
    hist = ver_proj <= th

    #get mean coordinates of white pixels groups
    x_coords = []
    x = 0
    count = 0
    is_space = False 

    for i in range(0, line.shape[1]):
        if not is_space:
            #when space has detected, get the first starting x_coordinate and start count at 1
            if hist[0][i]:
                is_space = True 
                count = 1
                x = i #coordinate of space
        else:
            if not hist[0][i]:
                is_space = False 
                #after smoothing, the thin letters may be breakdown 
                #so we create a new blank lines or pixel colums but the count will be small
                #then we set a threshold
                if (count > thres_space):
                    x_coords.append(x / count)
            else:
                x += i 
                count += 1
    x_coords.append(x / count)

    return x_coords



def get_spaces_median(line):
    #making vertical projections
    ver_proj = cv2.reduce(line, 0, cv2.REDUCE_AVG)

    #make hist: if 0 - mean space then set True else set the value False
    th = 0
    hist = ver_proj <= th

    #get mean coordinate of white pixels group
    x_coords = []
    x = 0
    count = 0
    is_space = False
    median_count = []

    #start count median space in line
    for i in range(0, line.shape[1]):
        if not is_space:
            #if space is detected then get the first starting x_coordinate and start count median at 1
            if hist[0][i]:
                is_space = True 
                count = 1
        else:
            if not hist[0][i]:
                is_space = False
                #after smoothing the letters in image will be breakdown, pay attention to fix this

                median_count.append(count)
            else:
                count += 1

    median_count.append(count)
    x_coords.append(x / count)
    #result of this function is the arrays of x_coordinates of spaces in the line
    return median_count


def get_spaces_threshold(y_coords, img_for_det):
    #find median for setting thredshold
    #median_list contains all the count of each blank founds in all lines
    #this is including spaces between each characters too
    median_list = []
    for i in range(0, len(y_coords) - 1):
        line = img_for_det[list(range(int(y_coords[i]), int(y_coords[i + 1])))]
        median_list.append(get_spaces_median(line))
    
    #find the row among median_list with max length
    max_len = len(median_list[0]) # init max length of the row in the list
    max_in = 0  # init index of the row has max_length
    for i in range(0, len(median_list)):
        if max_len < len(median_list[i]):
            max_len = len(median_list[i])
            max_in = i

    #sort the row having the maximum number of elements // decending order
    m_list = sorted(median_list[max_in], reverse=True)    
    #delete elements produced from the page margin
    m_list = np.delete(m_list, [0, 1])

    first_ele = m_list[0]
    # for i in range(len(m_list) - 1, 0, -1):
    #     if m_list[i] < first_ele / 2:
    #         m_list = np.delete(m_list, i)

    mean = np.mean(m_list)
    spaces_threshold = mean / 2

    return spaces_threshold    

def get_words(img_url, save_image=True):
    #get data from segmentation lines for segmentation words
    y_coords, img_for_det, img_for_ext = get_lines_segmentation(img_url)

    path_words = img_url[:-4] +'/words'

    #start segmentation words
    max_height_on_line = []
    for i in range(0, len(y_coords) - 1): # iterate lines
        line = img_for_det[range(int(y_coords[i]), int(y_coords[i + 1]))]

        #to find the max_line_height of each line we find contours again in this line only
        contour0, _ = cv2.findContours(line.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        


        max_line_height = 0
        #print max_line_height
        sorted_ctrs = sorted(contour0, key=lambda ctr: cv2.boundingRect(ctr)[0])
        for i, ctr in enumerate(sorted_ctrs):
            x, y, w, h = cv2.boundingRect(ctr)
            if(h > max_line_height):
                max_line_height = h
        max_height_on_line.append(max_line_height)

        #detection space in a line (prepare for a segmentation words)

        #get the threshold to dertermine how much gap should be considered as the space between the words
    threshold_space = get_spaces_threshold(y_coords, img_for_det)

    #split lines base on the y_coordinates of the detected lines
    #each line is put into the var line and the words are found 
    #base on the threshold_space value

    words_on_line = []
    all_words = [] #array of the segmentation words in line
    count = 0
    number_of_words = 0

    for i in range(0, len(y_coords) - 1): #iterate line
        #save lines image
        line = img_for_det[range(int(y_coords[i]), int(y_coords[i + 1]))]
        line_ext = img_for_ext[range(int(y_coords[i]), int(y_coords[i + 1]))]
        cv2.imwrite(img_url[:-4] + '/' + str(i) + '.png', line_ext)

        #finding the x_coordinates of the spaces
        x_coords = get_spaces(line, threshold_space)
        #save segmentation line
        for x in x_coords:
            x = int(x)
            cv2.line(line_ext, (x, 0), (x, line.shape[0]), 0, 1)
        cv2.imwrite(img_url[:-4] + '/' + str(i) + '.png', line_ext)

        count = 0

        for j in range(0, len(x_coords) - 1):
            #use the image with no smoothing
            line = img_for_det[list(range(int(y_coords[i]), int(y_coords[i + 1])))]
            word = line[:, int(x_coords[j]): int(x_coords[j + 1])]
            all_words.append(word)

            if save_image:
            #save segmentation words 
                line_ext = img_for_ext[list(range(int(y_coords[i]), int(y_coords[i + 1])))]
                word_ext = line_ext[:, int(x_coords[j]): int(x_coords[j + 1])]
                cv2.imwrite(path_words + '/' + str(number_of_words) + '.png', word_ext)

            count += 1
            number_of_words += 1

        words_on_line.append(count)
    return all_words, words_on_line, max_height_on_line

def get_words_segmentation(img_url, save_image=True):
    all_words, words_on_line, max_height_on_line = get_words(img_url, save_image)

    return all_words, words_on_line, max_height_on_line
    

            
