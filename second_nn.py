import numpy as np
from network import CrossEntropyCost, Network

def get_eq_let_ijl1I(number):
    if number == 0:
        return 18
    if number == 1:
        return 19
    if number == 2:
        return 21
    if number == 3:
        return 29
    if number == 4:
        return 44
    if number == 5:
        return 47
    if number == 6:
        return 1

def get_eq_let_ceg(number):
    if number == 0:
        return 12
    if number == 1:
        return 14
    if number == 2:
        return 42

def get_let_from_2nd_nn_ijltIL1(letter):
    net = Network([1024, 30, 7], cost=CrossEntropyCost)

    biases_saved = np.load("./training_model/biases_ijltIL1.npy", encoding='latin1', allow_pickle=True)
    weights_saved = np.load("./training_model/weights_ijltIL1.npy", encoding='latin1', allow_pickle=True)

    output = np.argmax(net.feedforward(letter, biases_saved=biases_saved, weights_saved=weights_saved))

    return get_eq_let_ijl1I(output)

def get_let_from_2nd_nn_ceg(letter):
    net = Network([1024, 30, 3], cost=CrossEntropyCost)
    
    biases_saved = np.load("./training_model/biases_ceg.npy", encoding='latin1', allow_pickle=True)
    weights_saved = np.load("./training_model/weights_ceg.npy", encoding='latin1', allow_pickle=True)

    output = np.argmax(net.feedforward(letter, biases_saved=biases_saved, weights_saved=weights_saved))

    return get_eq_let_ceg(output)