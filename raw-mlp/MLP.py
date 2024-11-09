from Matrix import Matrix
import random
import math

INPUT_NUM = 2

def add_bias(dot_mat: Matrix, bias_v: Matrix) -> Matrix:
        z = dot_mat
        for i in range(z.n_cols):
            for j in range(z.n_rows):
                z.matrix[j][i] += bias_v.matrix[i][0]

        return z

def sigmoid(x):
    return 1 / (1 + math.e**(-x))

def convert_labels(y: Matrix):
    one_hot_data = Matrix(y.n_rows, 10)
    for i in range(y.n_rows):
        one_hot = [0 for _ in range(10)]
        one_hot[y.matrix[i][0]] = 1
        one_hot_data.matrix[i] = one_hot


class MLP:
    def __init__(self, hidn_lay_num, hidn_node_num, ouput_num, l_rate):
        self.hidn_lay_num = hidn_lay_num
        self.hidn_node_num = hidn_node_num
        self.output_num = ouput_num
        self.weights = []
        self.bias = []
        self.l_rate = l_rate
        self.a_ = []

        self.initialize_weights_bias(layer_type=0) # initializing the first hidden layer
        for _ in range(self.hidn_lay_num-1): # initializing every other hidden layer
            self.initialize_weights_bias(layer_type=1)
        self.initialize_weights_bias(layer_type=2) # initializing the output layer

     # Initilize the weights and bias of the first hidden layer
    def initialize_weights_bias(self, layer_type=1):
        if layer_type == 0: # first layer
            layer_weights = Matrix(self.hidn_node_num, INPUT_NUM)
            layer_bias = Matrix(self.hidn_node_num, 1)
        elif layer_type == 1: # any other hidden layer
            layer_weights = Matrix(self.hidn_node_num, self.hidn_node_num)
            layer_bias = Matrix(self.hidn_node_num, 1)
        else: # output layer
            layer_weights = Matrix(self.output_num, self.hidn_node_num)
            layer_bias = Matrix(self.output_num, 1)
            
        layer_weights.element_oper(lambda element : random.uniform(-1, 1))
        self.weights.append(layer_weights)
                
        layer_bias.element_oper(lambda element : random.uniform(-1, 1))
        self.bias.append(layer_bias)


    def feed_forward(self, X):
        
        dot_mat = Matrix.dot(X, self.weights[0].mat_t(return_mat=True))
        z = add_bias(dot_mat, self.bias[0])
        z.element_oper(sigmoid)
        self.a_.append(z)

        for i in range(1, len(self.weights)):
            dot_mat = Matrix.dot(z, self.weights[i].mat_t(return_mat=True))
            z = add_bias(dot_mat, self.bias[i])
            z.element_oper(sigmoid)
            self.a_.append(z)

        # TODO: calculate the cost for each neuron
        cost_mat = Matrix(1, 1)

        return cost_mat
    
    def backwards_propagation(self, cost_mat: Matrix, y):

        '''
        z = sigma(x_i * w_i) + b

        L = sigma(y - a)^2

        
        
        for each weight of each output neuron:
        w = w - l_rate * (dL/dw) 
        dL/dw = dz/dw * da/dz * dL/da
        dz/dw = a_h
        da/dz = f'(z) = (1/1+e^(-z)) * (1 - 1/1+e^(-z)) 
        dL/da = 2(a-y)

        update to the weights of the output layer
        dL/dw = a(of the prev layer) * (1/1+e^(-z)) * (1 - 1/1+e^(-z)) * 2(a-y)


        '''


        # getting the gradient for updating the output weights
        
        
       

          
