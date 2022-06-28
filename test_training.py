# from sklearn.neural_network import MLPClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split

# X, y = make_classification(n_samples=100, random_state=1)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=1)
# print(X_train, X_test, y_train, y_test)
# clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
# print(clf.predict_proba(X_test[:1]))
# print(clf.predict(X_test[:5, :]))
# print(clf.score(X_test, y_test))

from distutils.command.config import config
from PIL import Image
import pytesseract
from pytesseract import Output
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'

def erode(image):
    kernel = np.ones((1,1),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#grayscale
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#thresholding
def thresholding(image):
    #return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    return cv2.threshold(image, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

def draw_contours(max_line_height):
    filter_image('./images/original_image/img_for_ext.png')
    image = cv2.imread('./images/original_image/img_for_ext.png')
    image = get_grayscale(image)
    image = cv2.GaussianBlur(image, (1, 1), 0)
    image = thresholding(image)

    image1 = cv2.imread('./images/original_image/img_with_lines.png')
    image2 = cv2.imread('./images/original_image/img_with_lines.png')
    # image1 = image.copy()
    # image2 = image.copy()

    image = cv2.Laplacian(image, cv2.CV_8UC1, ksize=3)

    h, w = image.shape
    
    # custom_config = r'-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyz'
    custom_config = r'--psm 6'
    results = pytesseract.image_to_data(image, output_type=Output.DICT,lang='eng', config=custom_config)
    # boxresults = pytesseract.image_to_boxes(image, output_type=Output.DICT,lang='eng', config=custom_config)
    boxresults = pytesseract.image_to_boxes(image, output_type=Output.DICT,lang='eng', config=custom_config)
    # print(results)
    # print(boxresults)
    # lt = results['left'][0]
    # top = results["top"][0]
    # for i in range(0, len(results["text"])):
    #     # extract the bounding box coordinates of the text region from the current result
    #     left = results["left"][i]
    #     width = results["width"][i]
    #     if abs(left - lt) < 5:
    #         top = results["top"][i]
    #     # tmp_br_y = tmp_tl_y - max_line_height
    #     tmp_level = results["level"][i]
        
    #     if(tmp_level == 5):
    #         cv2.rectangle(image2, (left, h - top), (left + width, top + max_line_height), 0, 1)
    # print(results['text'])
    for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from the current result
        tmp_tl_x = results["left"][i]
        tmp_tl_y = results["top"][i]
        tmp_br_x = tmp_tl_x + results["width"][i]
        tmp_br_y = tmp_tl_y + results["height"][i] 
        tmp_level = results["level"][i]
        if(tmp_level == 5):
            cv2.rectangle(image2, (tmp_tl_x, tmp_tl_y), (tmp_br_x, tmp_br_y), (0, 0, 0), 1)
            
    cv2.imwrite("./images/image_with_contours_words.png", image2)
    num_of_characters = len(boxresults["left"])
    print('number of characters: ', num_of_characters)
    lt = boxresults['left'][0]
    top = boxresults["top"][0]
    for j in range(0, num_of_characters):
        left = boxresults["left"][j]
        # bottom = boxresults["bottom"][j]
        right = boxresults["right"][j]
        if abs(left - lt) < (right - left):
            top = boxresults["top"][j]        
        cv2.rectangle(image1, (left, h - top - 3), (right, h - top + max_line_height), (0, 0, 0), 1)
        
    cv2.imwrite("./images/image_with_contours_characters.png", image1)

def reform_text(text):
    lines = text.split("\n")
    new_text = [line for line in lines if line.strip() != '']
    result = ''
    for line in new_text:
        result += line + '\n' 
    return result

def get_test(img_url='./images/original_image/img_for_ext.png', filename='./output.txt'):    
    image = cv2.imread(img_url)
    image = get_grayscale(image)
    image = cv2.GaussianBlur(image, (1, 1), 0)
    image = cv2.Laplacian(image, cv2.CV_8UC1, ksize=3)
    image = thresholding(image)
    
    config = r'--oem 1'
    text = reform_text(pytesseract.image_to_string(image, lang="eng", config=config))
    f = open(filename, 'w')
    f.write(text)
    f.close()
    # print(text)

def filter_image(img_url):
    image = cv2.imread(img_url)
    # image = get_grayscale(image)
    # image = cv2.GaussianBlur(image, (1, 1), 0)
    image = cv2.bitwise_not(cv2.Laplacian(image, cv2.CV_8U, ksize=5))
    # image = thresholding(image)
    # image = cv2.bitwise_not(image)
    cv2.imwrite('./images/image_filter.png', image)

# filter_image('./images/original_image/img_for_ext.png')
filter_image('./images/test_decorate_not_angel.png')
# draw_contours(54)