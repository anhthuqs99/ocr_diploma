from ctypes import resize
import sys 
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QWidget, QTabWidget, QFormLayout, QLabel, QScrollArea, QGridLayout, QVBoxLayout, QPushButton
from PyQt5.QtGui import QFont
# from crop_image import crop_image_shape 
from PIL import Image

import cv2
import os
from ocr import perform_ocr 

IMG_PATH = './images/'

class Window(QtWidgets.QMainWindow):
    location_qlable_signal = pyqtSignal()

    def __init__(self):
        super(Window, self).__init__()
        self.setGeometry(50, 50, 1400, 740)
        self.setWindowTitle("OCR system")
        QtWidgets.QApplication.setStyle(QtWidgets.QStyleFactory.create("Plastique"))

        #========================= Menu Bar ============================
        extractAction = QtWidgets.QAction("&Exit program", self)
        extractAction.setShortcut("Ctrl+Q")
        extractAction.setStatusTip("Quit Program")
        extractAction.triggered.connect(self.close_app)

        self.statusBar()

        #============Main screen with Tabs=====================
        self.tab_originals = Tab(self)
        self.tab_rotations = Tab(self)
        self.tab_binarizes = Tab(self)
        self.tab_filterses = Tab(self)
        self.tab_seg_lines = Tab(self)
        self.tab_seg_words = Tab(self)
        self.tab_seg_chars = Tab(self)
        self.tab_results = Tab(self)

        # allocate tab screen
        self.tabs = QTabWidget(self)
        self.tabs.setGeometry(QtCore.QRect(30, 30, 1150, 600))
        
        self.tabs.addTab(self.tab_originals, 'Original Image')
        self.tabs.addTab(self.tab_rotations, 'Rotated Image')
        self.tabs.addTab(self.tab_binarizes, 'Binarized Image')
        self.tabs.addTab(self.tab_filterses, 'Filtered Image')
        # self.tabs.addTab(self.tab_seg_lines, 'Segmented lines')
        # self.tabs.addTab(self.tab_seg_words, 'Segmented words')
        self.tabs.addTab(self.tab_seg_chars, 'Segmented Image')
        self.tabs.addTab(self.tab_results, 'Result')
        
        # self.tabs.setGeometry(QtCore.QRect(30, 30, 1100, 600))
        

        #============Button Menu===========================
        # load image
        self.load_image_button = QPushButton('Load image', self)
        self.load_image_button.clicked.connect(self.load_image)
        self.load_image_button.setStatusTip("Loading image...")

        #extract text
        self.extract_text_button = QPushButton('Extract text', self)
        self.extract_text_button.clicked.connect(self.extract_text)
        self.extract_text_button.setStatusTip('Extracting image...')

        #open file result
        self.open_file_button = QPushButton('Open result', self)
        self.open_file_button.clicked.connect(self.open_file_output)
        self.open_file_button.setStatusTip('Open file result...')

        # Allocate button 
        self.vertical_button_box = QVBoxLayout()
        self.vertical_button_box.addWidget(self.load_image_button)
        self.vertical_button_box.addWidget(self.extract_text_button)
        self.vertical_button_box.addWidget(self.open_file_button)
        self.vertical_button_box.setGeometry(QtCore.QRect(1210, 175, 150, 450))

        self.show()

    def load_image(self):
        name = QtWidgets.QFileDialog.getOpenFileName(self, "Select image", IMG_PATH, "Image files (*.jpg *.png)")
        file_name = str(name[0])
        self.location_qlable_signal.emit()

        input_image = cv2.imread(file_name)
        if input_image is not None:
            cv2.imwrite(IMG_PATH + "original_image.png", input_image) #save image for debug
            print("Image Load Complete...!")
        else:
            return 
        self.tab_originals.update("original_image.png")
    
    def extract_text(self):
        print("Extracting image...")
        perform_ocr(IMG_PATH + "original_image.png")

        self.tab_rotations.update("original_image/img_rotated.png")
        self.tab_filterses.update("original_image/img_for_ext.png")
        self.tab_binarizes.update("image_filter.png")
        self.tab_seg_lines.update("original_image/img_with_lines.png")
        self.tab_seg_words.update("image_with_contours_words.png")
        self.tab_seg_chars.update("image_with_contours_characters.png")
        text = open('./output.txt', 'r').read()
        self.tab_results.show_text(text)
        print("Extract text from image completed...!")

    def close_app(self):
        choice = QtWidgets.QMessageBox.question(self, "Exit", "Are you sure you want to quit?", QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        if choice == QtWidgets.QMessageBox.Yes:
            print("Quit program!")
            sys.exit()
        else:
            pass
    def open_file_output(self):
        os.startfile("output.txt")


class Tab(QWidget):
    def __init__(self, parent=None):
        super(Tab, self).__init__(parent)
        self.layout = QGridLayout()
        self.label = QtWidgets.QLabel()
        image = cv2.imread(IMG_PATH + 'default.png')
        resized_image = self.resize_image(image, 1024, 576)
        (res_height, res_width) = resized_image.shape[:2] 
        q_image = QtGui.QImage(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), res_width, res_height, res_width*3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(q_image)
        
        self.label.resize(1100, 600)
        self.label.setPixmap(pixmap.scaled(self.label.size(), QtCore.Qt.KeepAspectRatio))
        self.layout.addWidget(self.label)
        self.label.setParent(None)
        self.setLayout(self.layout)
        
    
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
        img = cv2.resize(img, (width, height))
        return img
    
    def update(self, image_url):
        new_label = QtWidgets.QLabel()
        image = cv2.imread(IMG_PATH + image_url)
        resized_image = self.resize_image(image, 1024, 576)
        (res_height, res_width) = resized_image.shape[:2] 
        q_image = QtGui.QImage(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB), res_width, res_height, res_width*3, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap(q_image)
        new_label.resize(1100, 600)
        new_label.setPixmap(pixmap.scaled(new_label.size(), QtCore.Qt.KeepAspectRatio))
        self.layout.replaceWidget(self.label, new_label)
        self.label.setParent(None)
        # self.label.deleteLater()
        self.show()
    
    def show_text(self, text):
        new_label = QtWidgets.QTextEdit()
        new_label.setFont(QFont('Arial', 20))
        new_label.setPlainText(text)
        new_label.resize(1100, 600)
        self.layout.replaceWidget(self.label, new_label)
        self.label.setParent(None)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    GUI = Window()
    sys.exit(app.exec_())