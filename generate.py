from PIL import Image, ImageDraw, ImageFont, ImageOps
import numpy as np
import os
from get_equivalent_letter import get_letter

import glob


WIDTH = 32
HEIGHT = 32
IMG_PATH = './images/generate_characters/'
FONT_PATH = './images/fonts/'
DATA_PATH = './training_model/'

def show_image_np_array(img):
    img_array = np.asarray(img)

    for line in img_array:
        # print(line)
        for c in line:
            print(f'{c:4d}', end='')
        print()

def create_test_image():
    font = './images/fonts/UnifrakturMaguntia-Regular.ttf'
    # text = 'Even or odd, of all days in the near,\nCome Lammas-eve at night shall she be fourteen.'
    # text = 'abcdefghijklmnopqrstuvwxyz'
    text = open('./original_text.txt', 'r').read()
    # text = text.upper()
    width = 1500
    height = 1000
    image = Image.new('L', (width, height), color=239)
    # new_image = Image.new('L', (width, height), color=255)
    font = ImageFont.truetype(font, 60)
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(text, font=font)
    drawing.text(((width - w) / 2, (height - h) / 2), text, fill=109, font=font, spacing=27)
    image = image.rotate(3, expand=1)
    image.save('./images/test_decorate.png', 'PNG')
    return 

def generate_image():
    fonts = glob.glob(os.path.join(FONT_PATH, '*.ttf')) 
    total_count = 0
    for i in range(66):
        character = get_letter(i)
        for font in fonts:
            total_count += 1
            image = Image.new('L', (WIDTH, HEIGHT), color=0)
            font = ImageFont.truetype(font, 28)
            drawing = ImageDraw.Draw(image)
            w, h = drawing.textsize(character, font=font)
            drawing.text(((WIDTH - w) / 2, (HEIGHT - h) / 2), character, fill=(255), font=font)
            file_name = str(i) + '_' + str(total_count) + '.png'
            file_path = IMG_PATH + file_name
            show_image_np_array(image)
            image.save(file_path, 'PNG')

    # return 

def generate_data_training():
    fonts = glob.glob(os.path.join(FONT_PATH, '*.ttf')) 
    total_count = 0
    datas = []
    for i in range(66):
        character = get_letter(i)
        file_path = IMG_PATH + str(i) + './'
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        else:
            filelist = glob.glob(os.path.join(file_path, "*"))
            for f in filelist:
                os.remove(f)
        count = 0
        for _ in range(100):
            for f in fonts:
                for size in range(20, 24):
                    image = Image.new('L', (WIDTH, HEIGHT), color=0)
                    font = ImageFont.truetype(f, size)
                    drawing = ImageDraw.Draw(image)
                    w, h = drawing.textsize(character, font=font)
                    total_count += 1
                    count += 1
                    drawing.text((abs((WIDTH-w)/2), abs((HEIGHT-h)/2)), character, fill=(255), font=font)
                    file_name = str(i) + '_' + str(count) + '.png'
                    # show_image_np_array(image)
                    image_array = np.asarray(image)
                    data = {'img':[], 'label':''}
                    data['img'] = image_array
                    data['label'] = str(i)
                    datas.append(data)
                    image.save(file_path + file_name, 'PNG')
    # print(datas)
    print('total training images created: ', total_count)
    return datas

def generate_data_validation():
    fonts = glob.glob(os.path.join(FONT_PATH, '*.ttf')) 
    total_count = 0
    datas = []
    for i in range(66):
        character = get_letter(i)
        # for _ in range(66):
        for size in range(20, 24):
            for f in fonts:
                image = Image.new('L', (WIDTH, HEIGHT), color=0)
                font = ImageFont.truetype(f, size)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                # for w in range((WIDTH - width) // 2):
                #     for h in range((HEIGHT - height) // 2):
                total_count += 1
                drawing.text(((WIDTH-w)/2, (HEIGHT-h)/2), character, fill=(255), font=font)
                file_name = str(i) + '_' + str(total_count) + '.png'
                file_path = IMG_PATH + file_name
                # show_image_np_array(image)
                image_array = np.asarray(image)
                data = {'img':[], 'label':''}
                data['img'] = image_array
                data['label'] = str(i)
                datas.append(data)
                # image.save(file_path, 'PNG')
    # print(datas)
    print('total validation images created: ', total_count)
    return datas

def generate_data_test():
    fonts = glob.glob(os.path.join(FONT_PATH, '*.ttf')) 
    total_count = 0
    datas = []
    for i in range(66):
        character = get_letter(i)
        # for _ in range(66):
        for size in range(20, 24):
            for f in fonts:
                image = Image.new('L', (WIDTH, HEIGHT), color=0)
                font = ImageFont.truetype(f, size)
                drawing = ImageDraw.Draw(image)
                w, h = drawing.textsize(character, font=font)
                # for w in range((WIDTH - width) // 2):
                #     for h in range((HEIGHT - height) // 2):
                total_count += 1
                drawing.text(((WIDTH-w)/2, (HEIGHT-h)/2), character, fill=(255), font=font)
                file_name = str(i) + '_' + str(total_count) + '.png'
                file_path = IMG_PATH + file_name
                # show_image_np_array(image)
                image_array = np.asarray(image)
                data = {'img':[], 'label':''}
                data['img'] = image_array
                data['label'] = str(i)
                datas.append(data)
                # image.save(file_path, 'PNG')
    # print(datas)
    print('total testing images created: ', total_count)
    return datas

def save_data():
    datas_training = generate_data_training()
    datas_validation = generate_data_validation()
    datas_testing = generate_data_test()
    datas_set = {'training_data': [], 'validation_data': [], 'testing_data': []}
    datas_set['training_data'] = datas_training
    datas_set['validation_data'] = datas_validation
    datas_set['testing_data'] = datas_testing
    file_path = DATA_PATH + 'data_set_264'
    np.savez(file_path, training_data=datas_set['training_data'], validation_data=datas_set['validation_data'], testing_data=datas_set['testing_data'])

def create_image_time_run():
    font = './images/fonts/UnifrakturMaguntia-Regular.ttf'
    # text = 'Even or odd, of all days in the near,\nCome Lammas-eve at night shall she be fourteen.'
    # text = 'abcdefghijklmnopqrstuvwxyz'
    text = open('./time_run.txt', 'r').read()
    # text = text.upper()
    width = 1500
    height = 10000
    image = Image.new('L', (width, height), color=239)
    # new_image = Image.new('L', (width, height), color=255)
    font = ImageFont.truetype(font, 60)
    drawing = ImageDraw.Draw(image)
    w, h = drawing.textsize(text, font=font)
    drawing.text(((width - w) / 2, (height - h) / 2), text, fill=109, font=font, spacing=27)
    image.save('./images/test_time_run_5000.png', 'PNG')
    return 

# generate_image()
# generate_data()
# create_test_image()
create_image_time_run()
# save_data()
# generate_data_training()

