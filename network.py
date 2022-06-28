import json 
import random
import sys 
import warnings

warnings.filterwarnings('ignore')

import numpy as np

# The sigmoid function (activation function)
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))
# derivative of the sigmoid function
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

#return a 10_dimentional unit vector with a 1.0 in the j-th position and reros elsewhere
# use to convert a digit into a cooresponding desires output from the neural network
def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    
    return e

# define the quadratic function 
class QuadraticCost(object):
    #Get the cost associated with an output a and desired the output (loss function)
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a-y)**2
    #return the error delta from the output layer
    @staticmethod
    def delta(z, a, y):
        return (a-y) * sigmoid_prime(z)

#define the cross-entropy cost function
class CrossEntropyCost(object):
    #return the cost associated with an output a and desired output y 
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    #return the error delta from the output layer
    @staticmethod
    def delta(z, a, y):
        return (a - y)

#network class
class Network(object):
    # the biases and weights for the network are initialized randomly using function default_weight_initializer
    def __init__(self, sizes, cost=CrossEntropyCost):
        self.num_layers = len(sizes)
        self.sizes = sizes # sizes contains the number of neurons in the respective layers of the network
        self.default_weight_initializer()
        self.cost = cost
    # initialize each weight using a Gaussian distribution with mean 0 and standard deviation 1
    def default_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y, x) / np.sqrt(x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def large_weight_initializer(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights  =[np.random.randn(y, x) for x, y in zip(self.sizes[:-1], self.sizes[1:])]

    def feedforward(self, a, biases_saved = None, weights_saved = None):
        # if the biases and weights are both supplied -> saved it 
        if biases_saved is not None and weights_saved is not None:
            self.biases = biases_saved
            self.weights = weights_saved
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b)
        # return the output of the network if a is input
        return a

    #update the network weights and biases by appling gradient descent using backpropagation to a single mini batch
    def update_mini_batch(self,
                          mini_batch, # list of tuples (x, y)
                          eta, # learning rate
                          lmbda, # regularization parameter
                          n): # total size of the training data set
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [(1 - eta*(lmbda/n))*w - (eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [n - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]


    # retun the number of inputs in data for which the neural network outputs the correct result
    # the neural network's output is assumed to be the index of whichever neuton in the final layer has the highest activation
    def accuracy(self, data, convert=False): # flag convert=False if the data set is validation or test data, convert=True when training data
        if convert:
            result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
            return sum(int(x == y) for (x, y) in result)
        else:
            result = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
            return sum(int(x == sum(y)) for (x, y) in result)

            # result = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
            # return sum(int(x == y) for (x, y) in result)
        
    
    # return the total cost for the data set data
    def total_cost(self, data, lmbda, convert=False): #flag convert=False if the data set is a training data, convert=True if data set is validation or test data
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert:
                y = vectorized_result(np.argmax(y))
            cost += self.cost.fn(a, np.argmax(y)) / len(data)
        cost += 0.5*(lmbda/len(data))*sum(np.linalg.norm(w)**2 for w in self.weights)

        return cost

    # train the neural network using mini-batch stochastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta,
            lmbda = 0.0,
            evaluation_data = None,
            monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False, 
            monitor_training_cost = False,
            monitor_training_accuracy = False):
        
        # print(self.biases)
        # print(self.weights)
        n_data = 0
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        print('length of training data: ', n)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            print("minibatches len: ", len(mini_batches))
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmbda, len(training_data))
            print("Epoch %s training complete" %j)
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmbda)
                training_cost.append(cost)
                print("cost on training data: {}".format(cost))
            if monitor_training_accuracy:
                accuracy = self.accuracy(training_data, convert=True)
                training_accuracy.append(accuracy)
                print("accuracy on training data: {} / {}".format(accuracy, n))
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmbda, convert=True)
                evaluation_cost.append(cost)
                print("cost on evaluation data: {}".format(cost))
            if monitor_evaluation_accuracy:
                accuracy = self.accuracy(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print("accuracy on evaluation data: {} / {} - {}".format(self.accuracy(evaluation_data), n_data, self.accuracy(evaluation_data) / float(n_data) * 100))
            np.save('./training_model/biases' + str(round(self.accuracy(evaluation_data, convert=True) / float(n_data)*100, 3)), self.biases)
            np.save('./training_model/weights' + str(round(self.accuracy(evaluation_data, convert=True) / float(n_data)*100, 3)), self.weights)
            print("biases and weights had written to file with accuracy: ", round(self.accuracy(evaluation_data, convert=True) / float(n_data)*100, 3), round(self.accuracy(training_data, convert=True) / float(n_data)*100, 3))
            print()
        # print(self.biases)
        # print(self.weights)
        np.save('./training_model/new_biases', self.biases)
        np.save('./training_model/new_weights', self.weights)
            
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    #save the neural network to the file file_name
    def save(self, file_name):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__),}
        f = open(file_name, "w")
        json.dump(data, f)
        f.close()
    
    #return a tuple (nabla_b, nabla_w) representing the gradient for the cost function C_x
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases] # layer by layer list of numpy arrays similar as self.biases
        nabla_w = [np.zeros(w.shape) for w in self.weights] # layer by layer list of numpy arrays similar as self.weights

        #feedforward
        activation = x
        activations = [x] #list to store all the activations, layer by layer
        zs = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b 
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        
        #backward pass
        delta = (self.cost).delta(zs[-1], activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp 
            nabla_b[-l] = delta 
            nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())
        
        return (nabla_b, nabla_w)

# load a neural network from the file_name and return the instance of network
def load(file_name):
    f = open(file_name, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Network(data["size"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]

    return net
    

        

    
