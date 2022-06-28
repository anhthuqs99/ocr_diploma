from http.client import REQUEST_HEADER_FIELDS_TOO_LARGE
from select import select
from ssl import VerifyFlags
import sys 
from PyQt5 import QtGui, QtCore, QtWidgets
from crop_image import crop_image_shape 
from PIL import Image

import cv2
import os
from ocr import perform_ocr 

IMG_PATH = './images/'

class Window(QtWidgets.QMainWindow):
    
    location_qlable_signal = QtCore.pyqtSignal()
    upload_complete_signal = QtCore.pyqtSignal()
    crop_image_signale = QtCore.pyqtSignal()
    original_height = 0
    original_width = 0
    boundary_xy = (0, 0, 0 ,0)
    is_crop = False
    

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 1280, 720)
        self.setWindowTitle("OCR system")
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Plastique"))

        #========================= Menu Bar ============================
        extractAction = QtWidgets.QAction("&Exit program", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip("Quit Program")
        extractAction.triggered.connect(self.close_app)

        self.statusBar()
        self.home()

        mainMenu = self.menuBar()

        fileMenu = mainMenu.addMenu("&File")
        fileMenu.addAction(extractAction)

        #====================== Main Screen =========================

    def home(self):
        # Select image button
        select_image_button = QtWidgets.QPushButton("Select Image", self)
        select_image_button.clicked.connect(self.load_image)
        select_image_button.setStatusTip("Browse for image")
        select_image_button.resize(select_image_button.sizeHint())

        #crop image button 
        crop_image_button = QtWidgets.QPushButton("Crop Image", self)
        crop_image_button.clicked.connect(self.crop_image)
        crop_image_button.setStatusTip("Browse for image")
        crop_image_button.setEnabled(False)
        self.upload_complete_signal.connect(lambda: crop_image_button.setEnabled(True))

        # extraction image button 
        extraction_image_button = QtWidgets.QPushButton("Extract Text", self)
        extraction_image_button.setStatusTip("Extract image from text")
        extraction_image_button.resize(extraction_image_button.sizeHint())
        extraction_image_button.clicked.connect(self.extraction_text)

        #open result file button
        open_result_button = QtWidgets.QPushButton("Open Result", self)
        open_result_button.clicked.connect(self.open_file_output)

        # button box on the right side
        vertical_button_box = QtWidgets.QVBoxLayout()
        vertical_button_box.addWidget(select_image_button)
        vertical_button_box.addWidget(crop_image_button)
        vertical_button_box.addWidget(extraction_image_button)
        vertical_button_box.addWidget(open_result_button)
        vertical_button_box.setGeometry(QtCore.QRect(1064, 175, 150, 450))

        self.show()

    def extraction_text(self):
        if self.is_crop:
            self.extraction_text_from_image(IMG_PATH + "cropped_image.jpg")
        else:
            self.extraction_text_from_image(IMG_PATH + "original_image.jpg")
    
    def extraction_text_from_image(self, image_url):
        print("Extracting image...")
        print(self.boundary_xy)
        perform_ocr(image_url)
        print("Extraction complete...!")
        # os.startfile("output.txt")
    
    def open_file_output(self):
        os.startfile("output.txt")
    
    def close_app(self):
        choice = QtWidgets.QMessageBox.question(self, "Exit", "Are you sure you want to quit?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("Quit program!")
            sys.exit()
        else:
            pass
    
    def store_original_shape(self, shape):
        self.original_height, self.original_width = shape[0], shape[1]
        self.boundary_xy = (0, 0, shape[1], shape[0])
    
    def resize_image(self, img, w, h):
        input_height = img.shape[0]
        input_width = img.shape[1]
        aspect_ratio = input_width / (input_height * 1.0)

        width = w 
        height = h 

        if aspect_ratio < 16/9:
            width = int(height * aspect_ratio) 
        elif aspect_ratio > 16/9:
            height = int(width / aspect_ratio)
        else:
            width = h
            # height = w 
        img = cv2.resize(img, (width, height))
        return img
    
    def store_cropped_shape(self, x1, y1, x2, y2):
        self.boundary_xy = (x1, y1, x2, y2)
        img = Image.open(IMG_PATH + 'original_image.jpg')
        box = (x1, y1, x2, y2)
        img = img.crop(box)
        img.save(IMG_PATH + 'cropped_image.jpg')
        print("Image cropped...")
        self.crop_image_signale.emit()
        self.is_crop = True
    
    def crop_image(self):
        img_url = IMG_PATH + 'cropped_image.jpg'
        x1, y1, x2, y2, to_crop_width, to_crop_height = crop_image_shape(img_url)
        x1_final = x1 * self.original_width / to_crop_width
        y1_final = y1 * self.original_height / to_crop_height 
        x2_final = x2 * self.original_width / to_crop_width 
        y2_final = y2 * self.original_height / to_crop_height
        self.store_cropped_shape(x1_final, y1_final, x2_final, y2_final)

    def load_image(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", IMG_PATH, "Image files (*.jpg *.png)")
        file_name = name[0]
        self.location_name = file_name
        self.location_qlable_signal.emit()

        input_img = cv2.imread(str(file_name))
        if input_img is not None:
            cv2.imwrite(IMG_PATH + "original_image.jpg", input_img) #save image for debug
            print("Image Load Complete...!")
        else:
            return 

        self.store_original_shape(input_img.shape)
        self.is_crop = False 
        self.upload_complete_signal.emit()

        resized_img = self.resize_image(input_img, 1024, 576)
        cv2.imwrite(IMG_PATH + "cropped_image.jpg", resized_img) #save image for debug

        res_height, res_width = resized_img.shape[0], resized_img.shape[1]
        q_image = QtGui.QImage(cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB), res_width, res_height, res_width*3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(q_image)

        input_image_label = QtWidgets.QLabel(self)
        input_image_label.setGeometry(20, 120, 1024, 576)
        input_image_label.setPixmap(pixmap)
        input_image_label.show()
        

        self.upload_complete_signal.connect(input_image_label.hide)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())