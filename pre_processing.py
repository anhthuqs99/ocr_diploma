import cv2 
import numpy as np

IMG_LOCATION = './images/'

def image_for_detection(raw_image):
    #remove tiny noise by blurring
    sm_image = cv2.GaussianBlur(raw_image, (1, 1), 0)
    
    #binarize
    _, bw_image = cv2.threshold(sm_image, 200, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    #dilate
    # kernel = np.ones((2, 2), np.uint8)
    # bw_image = cv2.dilate(bw_image, kernel)

    return bw_image

def image_for_extraction(raw_image):
	
	raw_image = cv2.GaussianBlur(raw_image, (1, 1), 0)
	
	_, no_sm_bw_image = cv2.threshold(raw_image, 0, 255, cv2.THRESH_OTSU)
	
	return no_sm_bw_image

#get angle for rotation 
def get_angle(img):
    # convert RGB to BW
    #values of pixels from 0 to 255
    img_gray = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)
 
    # а теперь из серых тонов, сделаем изображение бинарным
    th_box = int(img_gray.shape[0] * 0.007) * 2 + 1
    img_bin_ = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, th_box, th_box)
 
    img_bin = img_bin_.copy()
    num_rows, num_cols = img_bin.shape[:2]
 
    best_zero, best_angle = None, 0
    # итеративно поворачиваем изображение на пол градуса
    for my_angle in range(-20, 21, 1):
        rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows /2 ), my_angle/2, 1)
        img_rotation = cv2.warpAffine(img_bin, rotation_matrix, (num_cols*2, num_rows*2),
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=255)
 
        img_01 = np.where(img_rotation > 127, 0, 1)
        sum_y = np.sum(img_01, axis=1)
        th_ = int(img_bin_.shape[0]*0.005)
        sum_y = np.where(sum_y < th_, 0, sum_y)
 
        num_zeros = sum_y.shape[0] - np.count_nonzero(sum_y)
 
        if best_zero is None:
            best_zero = num_zeros
            best_angle = my_angle
 
        # лучший поворот запоминаем
        if num_zeros > best_zero:
            best_zero = num_zeros
            best_angle = my_angle
 
    return best_angle * 0.5

'''
Input: blackwhite binarized image file: text: white, background: black
Output: Matrix for transformation, flag of the page layout
'''
def get_transformation_matrix(img, is_portrait=False):
    #find all the white pixels
    pts = np.empty([0, 0])
    pts = cv2.findNonZero(img)

    # get rotated rect of the white pixels
    rect = cv2.minAreaRect(pts)
    # print(rect)

    # draw rectagle contain text for rotated
    drawrect = img.copy()
    drawrect = cv2.cvtColor(drawrect, cv2.COLOR_GRAY2BGR)
    box = cv2.boxPoints(rect)
    box = np.int0(box)     # box contains 4 vertices of rotated rectangle
    cv2.drawContours(drawrect,[box],0,(0,0,255),1)
    cv2.imwrite(IMG_LOCATION + 'rotated_rect.png', drawrect) #save image for debug

    rect = list(rect)
    # print(rect)
    # if is_portrait:
    #     if (rect[1][0] < rect[1][1]): # rect.size.width > rect.size.height 
    #         temp = list(rect[1])
    #         temp[0], temp[1] = temp[1], temp[0] #swap
    #         rect[1] = tuple(temp)
    #         rect[2] = abs(90 - rect[2])
    if rect[2] > 45:
        rect[2] = rect[2] - 90
    #convert rect back to numpy/ tuple
    rect = np.asarray(rect)
    #get matirx for rotation image
    M = cv2.getRotationMatrix2D(rect[0], rect[2], 1.0)
    # print(M)
    
    return M

def rotate(image, M, border=0):
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]), borderValue=border)

def image_aspect(img):
    h, w = img.shape
    return h > w

def pre_processing_image(image_url):
    raw_image = cv2.imread(image_url, 0)
    # print(image_url)

    # is_landscape = False
    is_portrait = image_aspect(raw_image)
    img_for_ext = image_for_extraction(raw_image)
    img_for_det = image_for_detection(raw_image)

    M = get_transformation_matrix(img_for_det, is_portrait)
    img_for_det = rotate(img_for_det, M)
    img_for_ext = rotate(img_for_ext, M, border=255)
    img_rotated = rotate(raw_image, M)

    # print('save images for detection and extration')
    cv2.imwrite(image_url[:-4] + '/' + 'img_for_det.png', img_for_det)
    cv2.imwrite(image_url[:-4] + '/' + 'img_for_ext.png', img_for_ext)
    cv2.imwrite(image_url[:-4] + '/' + 'img_rotated.png', img_rotated)

    return img_for_det, img_for_ext



