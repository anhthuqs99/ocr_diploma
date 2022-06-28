import numpy as np
import cv2

from segmentation_words import get_words_segmentation

def fix_i_j(rect, max_line_height, max_w):
    j = 0
    i_dot_list = []

    for i in rect:
        x, y, w, h = i[0], i[1], i[2], i[3]

        #if the dot of i is the last element in the rect then the j+1 index will not work
        #so we put j-1 seeparately here
        if (j is len(rect) - 1) and (h < max_line_height / 3):
            if (h < (max_line_height / 3)) and (abs(rect[j - 1][0] + rect[j - 1][2] - (x + w)) < (max_w / 3.5)):
                #correct i
                rect[j - 1] = (rect[j - 1][0], rect[j - 1][1], rect[j - 1][2], rect[j - 1][3] + (rect[j - 1][1] - y))

                #rect[j - 1][1] turn into value of y
                rect[j - 1] = (rect[j - 1][0], y, rect[j - 1][2], rect[j - 1][3])
                #save dot 
                i_dot_list.append(j)
            elif (h < (max_line_height / 2.4)) and (y > (rect[j - 1][1] + rect[j - 1][3] / 3)):
                #
                rect[j] = [x, y - max_line_height / 2, w, h + max_line_height / 2]
        #if the dot of i is not the last element in the rect:
        else:
            if (h < (max_line_height / 3)) and (abs(rect[j - 1][0] + rect[j - 1][2] - (x + w)) < (max_w / 3.5)):
                #correct i
                rect[j + 1] = (rect[j + 1][0], rect[j + 1][1], rect[j + 1][2], rect[j + 1][3] + (rect[j + 1][1] - y))

                #rect[j + 1][1] turn into value y
                rect[j + 1] = (rect[j + 1][0], y, rect[j + 1][2], rect[j + 1][3])

                i_dot_list.append(j)
            
            elif (h < (max_line_height / 3)) and (abs(rect[j - 1][0] + rect[j - 1][2] - (x + w)) < (max_w / 3.5)):
                rect[j - 1] = (rect[j - 1][0], rect[j - 1][1], rect[j - 1][2], rect[j - 1][3] + (rect[j - 1][1] - y))

                rect[j - 1] = (rect[j - 1][0], y, rect[j - 1][2], rect[j - 1][3])
            elif (h < max_line_height / 2.4) and (y > (rect[j - 1][1] + rect[j - 1][3] / 3)):
                rect[j] = [x, y - (max_line_height / 2), w, h + (max_line_height / 2)]
        j += 1
    
    #delete the dots from rect array which belong to i and j
    # rect = np.delete(rect, i_dot_list, axis=0)

    return rect

def get_chracters(raw_image, max_line_height, line, word, URL='./images/characters/', save_image=False):
    #find contours
    raw_img = raw_image.copy()
    contour0, _ = cv2.findContours(raw_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # print(contour0)
    '''
    '''
    rect = []
    max_w = 0
    sorted_ctrs = sorted(contour0, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        if h > max_w:
            max_w = h 
        rect.append(cv2.boundingRect(ctr))
    #there are two contours detected for the characters such as i and j
    #so we need to merge two contours of the dot and base
    rect = fix_i_j(rect, max_line_height, max_w)

    # print(rect)
    #remove artifacts
    #usually artifacts found are manpulated as 0 height by the i and j dot fixing fuctions 
    minus_count = 0
    minus_list = []

    for i in rect:
        (x, y, w, h) = i
        if h < 0:
            minus_list.append(minus_count)
        minus_count += 1
    rect = np.delete(rect, minus_list, axis=0)

    rect_segmented_image = raw_img.copy()
    # symbols = []

    all_letters = []

    count = 0 #used for naming file
    for i in rect:

        (x, y, w, h) = i
        # p1 = (int(x), int(y))
        p1 = (int(x), 1)
        p2 = (int(x + w), int(y + max_line_height - 3))
        letter = raw_img[int(y): int(y+h), int(x): int(x+w)]

        #resize letter image to 32*32 for detection 
        #resize letter content to 28*28 for detection 
        LETTER_SIZE = 26

        o_height = letter.shape[0]
        o_width = letter.shape[1]

        #if errors occurs due to the unwanted artifacts, then the height will somehow become zero
        if o_height == 0:
            letter = np.zeros((30, 30, 1), np.uint8)
            o_height = letter.shape[0]
            o_width = letter.shape[1]

        #resize height adn width to 28 pixels
        #we need three different conditions to work with the aspect ratios
        if o_height > o_width:
            aspect_ratio = o_width / (o_height * 1.0)
            height = LETTER_SIZE
            width = int(height * aspect_ratio)
            if width == 0:
                width = LETTER_SIZE
            letter = cv2.resize(letter, (width, height))

        elif o_height < o_width:
            aspect_ratio = o_height / (o_width * 1.0)
            width = LETTER_SIZE
            height = int(width * aspect_ratio)
            if height == 0:
                height = width
            letter = cv2.resize(letter, (width, height))
        else:
            letter = cv2.resize(letter, (LETTER_SIZE, LETTER_SIZE))

        #add border which results adding of padding
        #padding width
        remaining_pixels_w = abs(32 - letter.shape[1])
        add_left = int(remaining_pixels_w / 2)
        add_right = remaining_pixels_w - add_left 
        letter = cv2.copyMakeBorder(letter, 0, 0, add_left, add_right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        #padding height
        remaining_pixels_h = abs(32 - letter.shape[0])
        add_top = int(remaining_pixels_h / 2)
        add_bottom = remaining_pixels_h - add_top 
        letter = cv2.copyMakeBorder(letter, add_top, add_bottom, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        #save image character for debug
        if save_image:
            img_path = URL + str(line) + '_' + str(word) + '_' + str(count) + '.png'
            cv2.imwrite(img_path, letter)

        count += 1

        letter = letter / 255.0

        letter = np.reshape(letter, (1024, 1))

        all_letters.append(letter)
        if save_image:
            cv2.rectangle(rect_segmented_image, p1, p2, 255, 1)
    if save_image:
        cv2.imwrite(URL + str(line) + '_' + str(word) + '_segmented.png', rect_segmented_image)

    return all_letters

def get_chracters_segmentation(img_url, save_image=False):
    URL = img_url[:-4] + '/characters/'
    print(URL)
    all_words, words_on_line, max_height_on_line = get_words_segmentation(img_url, save_image)
    print(max_height_on_line)
    print('With image: ', img_url)
    print('number of words on image: ', len(all_words))
    print('number of words on line: ', words_on_line)
    print('='*30)

    count = 0
    for i in range(0, len(words_on_line)):
        for j in range(0, words_on_line[i]):
            all_characters = get_chracters(all_words[count], max_height_on_line[i], i, j, URL)
            # print('chracter on line: ',all_characters)
            count += 1

    return all_characters