from cgi import test
import numpy as np
from network import CrossEntropyCost, Network

# return a 10-dimensional unit vector with a 1.0 in the y-th position and zeros elsewhere
# used to convert a digit into a corresponding desired output from the neural network 
def vectorized_result(y):
    e = np.zeros((66, 1))
    e[y] = 1.
    
    return e

def load_data(file_name='./training_model/data_set.npz'):
    # data = np.load('./training_model/synthadd.npz', allow_pickle=True)
    data = np.load(file_name, allow_pickle=True)
    print("data fields: ")
    for k in data.files:
        print(k)
    print("================")
    training_data = data['training_data']
    # validation_data = data['training_data1']
    # testing_data = []
    validation_data = data['validation_data']
    testing_data = data['testing_data']

    print("data lenght:")
    print(len(training_data), len(validation_data), len(testing_data))
    print("================")

    return (training_data, validation_data, testing_data)

def load_data_wrapper(file_name='./training_model/data_set.npz'):
    train_d, valid_d, test_d = load_data(file_name)

    # training_inputs = [np.reshape(x, (1024, 1)) for x in train_d]
    # training_results = [vectorized_result(y) for y in train_d]
    training_inputs = []
    training_results = []
    for data in train_d:
        try:
            img = data['img']
            label = int(data['label']) #net number of equivalent chracter
            training_inputs.append(np.reshape(img, (1024, 1)))
            training_results.append(vectorized_result(label))
        except: pass
    training_data = list(zip(training_inputs, training_results))
    # print(training_data)

    # print(valid_d)
    # validation_inputs = [np.reshape(x, (1024, 1)) for x in valid_d[0]]
    # validation_data = list(zip(validation_inputs, valid_d[1]))
    validation_inputs = []
    validation_results = []
    for data in valid_d:
        try:
            img = data['img']
            label = int(data['label'])
            validation_inputs.append(np.reshape(img, (1024, 1)))
            validation_results.append(vectorized_result(label))
        except: pass 
    validation_data = list(zip(validation_inputs, validation_results))

    # testing_inputs = [np.reshape(x, (1024, 1)) for x in test_d[0]]
    # testing_data = list(zip(testing_inputs, test_d[1]))

    testing_inputs = []
    testing_result = []
    for data in test_d:
        try:
            img = data['img']
            label = int(data['label'])
            testing_inputs.append(np.reshape(img, (1024, 1)))
            testing_result.append(vectorized_result(label))
        except: pass 
    testing_data = list(zip(testing_inputs, testing_result))

    return (training_data, validation_data, testing_data)

def training_nn():
    training_data, validation_data, testing_data = load_data_wrapper()
    # print(len(training_data))
    # print(len(validation_data))
    # print(len(testing_data))
    net = Network([1024, 30, 66], cost=CrossEntropyCost)
    net.large_weight_initializer()
    net.SGD(training_data, 30, 1, 0.5, lmbda=5.0, evaluation_data=training_data, monitor_evaluation_cost=False, monitor_evaluation_accuracy=True, monitor_training_cost=False, monitor_training_accuracy=True)

if __name__ == '__main__':
    training_nn()
