import cv2
import numpy as np
from network import CrossEntropyCost, Network
from second_nn import get_let_from_2nd_nn_ijltIL1, get_let_from_2nd_nn_ceg
from get_equivalent_letter import get_letter
from segmentation_chracters import get_chracters
from segmentation_words import get_words_segmentation
from dictionary import correction
from test_training import draw_contours, get_test
import os
import glob

def get_string_from_nn(all_characters):
    net = Network([1024, 30, 66], cost=CrossEntropyCost)

    biases_saved = np.load("./training_model/biases.npy", encoding='latin1', allow_pickle=True)
    # biases_saved = np.load("./training_model/new_biases.npy", encoding='latin1', allow_pickle=True)
    weights_saved = np.load("./training_model/weights.npy", encoding='latin1', allow_pickle=True)
    # weights_saved = np.load("./training_model/new_weights.npy", encoding='latin1', allow_pickle=True)

    word_string = ""

    i = 0
    for x in all_characters:
        output = np.argmax(net.feedforward(x, biases_saved=biases_saved, weights_saved=weights_saved))

        if (output in (18, 19, 21, 44, 47, 1)):
            output = get_let_from_2nd_nn_ijltIL1(x)
        elif (output in (12, 14, 42)):
            output = get_let_from_2nd_nn_ceg(x)
        
        word_string += get_letter(output)
        i += 1
    # print(word_string)
    return word_string


def perform_ocr(img_url, save_image=True):

    fp = open("output.txt", "w")
    # fp = open('./results/' + str(test_number) + ".txt", "w")
    fp.truncate()
    URL = img_url[:-4] + '/characters/' # url for save images for debugs
    use_dictionary = True #flag of use dictionary for correction word 
    # use_dictionary = False
    # save_image = True #flag of save image for debug

    all_words, words_on_line, max_height_on_line = get_words_segmentation(img_url)
    draw_contours(max(max_height_on_line))
    print('max height on line: ', max_height_on_line)
    print("number of word: ", len(all_words))
    print("number of words on line: ", words_on_line)
    print("=============================================")
    # all_characters = get_chracters_segmentation(img_url)

    if save_image:
        if not os.path.exists(URL):
            os.makedirs(URL)
        else:
            filelist = glob.glob(os.path.join(URL, "*"))
            for f in filelist:
                os.remove(f)

    count = 0

    for i in range(0, len(words_on_line)):
        for j in range(0, words_on_line[i]):
            all_characters = get_chracters(all_words[count], max_height_on_line[i], i, j, URL=URL, save_image=save_image)
            # print(all_characters)
            if use_dictionary:
                fp.write(correction(get_string_from_nn(all_characters)))
            else:
                fp.write(get_string_from_nn(all_characters))
            fp.write(" ")
            count += 1
        fp.write("\n")

    fp.close()
    get_test(img_url)
    # get_test()

    
if __name__ == "__main__":
    perform_ocr("./images/test10.png")
