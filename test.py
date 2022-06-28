from ocr import perform_ocr
from pre_processing import image_for_detection, pre_processing_image
from segmentation_chracters import get_chracters_segmentation
from segmentation_lines import get_lines_segmentation
from segmentation_words import get_words
import os
import time

IMG_LOCATION = './images/'

def test_pre_processing_image():
    for i in range(1, 6):
        pre_processing_image(IMG_LOCATION + 'test' + str(i) + '.png')

def test_segmentation_words():
    for i in range(1, 3):
        path = IMG_LOCATION + 'test' + str(i)
        img_url = path + '.png'
        # print(img_url)
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(img_url):
            get_words(img_url)
def test_segmentation_lines():
    for i in range(1, 6):
        path = IMG_LOCATION + 'test' + str(i)
        img_url = path + '.png'
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(img_url):
            get_lines_segmentation(img_url)
def test_segmentation_characters():
    for i in range(1, 4):
        path = IMG_LOCATION + 'test' + str(i)
        path_characters = path + '/characters'
        img_url = path + '.png'
        # if os.path.exists(path):
        #     os.rmdir(path)
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path_characters):
            os.makedirs(path_characters)
        if os.path.exists(img_url):
            get_chracters_segmentation(img_url)

def test_ocr():
    for i in range(1, 7):
        path = IMG_LOCATION + 'test7_' + str(i) + '.jpg'
        print(path)
        perform_ocr(path, i)

def matching_2_words(word1, word2):
    count = 0
    j = 0
    for c in word1:
        if word2.find(c) and j == word1.find(c):
            count += 1
        j += 1
    return count

def compare_2_files(filename_1, filename_2):
    f1 = open(filename_1, 'r')
    f2 = open(filename_2, 'r')

    total_count = 0
    count = 0
    for line1, line2 in zip(f1, f2):
        for word1, word2 in zip(line1.split(), line2.split()):
            print(word1, word2)
            total_count += len(word1)
            count += matching_2_words(word1, word2)
    # for char_1, char_2 in zip(f1, f2):
    #     if char_1 != char_2:
    #         count += 1
    return total_count,  count

def compare_results():
    PATH = './results/'
    path_resource = PATH + '0.txt'
    for i in range(1, 2):
        path = PATH + str(i) + '.txt'
        print(compare_2_files(path_resource, path))

def time_run():
    start = time.time()
    perform_ocr('./images/test_time_run_100.png', save_image=False)
    stop = time.time()
    print('Time 100: ', stop - start)
    
    start = time.time()
    perform_ocr('./images/test_time_run_500.png', save_image=False)
    stop = time.time()
    print('Time 500: ', stop - start)
    
    start = time.time()
    perform_ocr('./images/test_time_run_1000.png', save_image=False)
    stop = time.time()
    print('Time 1000: ', stop - start) 
    
    start = time.time()
    perform_ocr('./images/test_time_run_2000.png', save_image=False)
    stop = time.time()
    print('Time 2000: ', stop - start) 
    
    start = time.time()
    perform_ocr('./images/test_time_run_5000.png', save_image=False)
    stop = time.time()
    print('Time 5000: ', stop - start) 


if __name__ == "__main__":
    # test_pre_processing_image()
    # test_segmentation_words()
    # test_segmentation_lines()
    # test_segmentation_characters()
    # test_ocr()
    # compare_results()
    time_run()