from PIL import Image
import numpy as np
from network import Network 
from training_nn import load_data_wrapper

IMG_PATH = './images/'
DATA_PATH = './training_model/'

def print_training():
    # train_d, valid_d, test_d = load_data_wrapper(DATA_PATH + 'new_data_set.npz')
    train_d, valid_d, test_d = load_data_wrapper()
    # for i in range(len(train_d)):
    #     # print(train_d[i])

    #     img = train_d[i]['img']
    #     label = int(train_d[i]['label'])
    #     # out_img = Image.fromarray(img)
    #     # out_img.save(IMG_PATH + label + '.png')
    #     print(label)
    #     for line in img:
    #         # print(line)
    #         for c in line:
    #             print(f'{c:4d}', end='')
    #         print()

def training():
    training_data, validation_data, testing_data = load_data_wrapper(DATA_PATH + 'data_set_264.npz')
    # training_data, validation_data, testing_data = load_data_wrapper()
    net = Network([1024, 30, 66])
    # net.large_weight_initializer()
    biases_saved = np.load("./training_model/biases.npy", encoding='latin1', allow_pickle=True)
    weights_saved = np.load("./training_model/weights.npy", encoding='latin1', allow_pickle=True)
    net.biases = biases_saved
    net.weights = weights_saved
    net.SGD(training_data=training_data, 
            epochs=30, 
            mini_batch_size=10, 
            eta=0.5, 
            lmbda=5.0, 
            evaluation_data=validation_data,
            monitor_evaluation_cost=False, 
            monitor_evaluation_accuracy=True, 
            monitor_training_cost=False, 
            monitor_training_accuracy=True)


    # print('evaluation cost: ', evaluation_cost)
    # print('evaluation accuracy: ', evaluation_accuracy)
    # print('training cost: ', training_cost)
    # print('training accuracy: ', training_accuracy)

    return 

def print_fields():
    bias = np.load('./training_model2/biases.npy', allow_pickle=True, encoding='latin1')
    weight = np.load('./training_model2/weights.npy', allow_pickle=True, encoding='latin1')

    print(bias)
    print(weight)

    # for k in bias.files:
    #     print(k)
    # print('======================')
    # for k in weight.files:
    #     print(k)

    return
# print_training()
training()
# print_fields()