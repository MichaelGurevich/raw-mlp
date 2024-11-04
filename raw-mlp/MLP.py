from Matrix import Matrix
import random

INPUT_NUM = 2

def add_bias(dot_mat: Matrix, bias_v: Matrix) -> Matrix:
        z = dot_mat
        for i in range(z.n_cols):
            for j in range(z.n_rows):
                z.matrix[j][i] += bias_v.matrix[i][0]

        return z


class MLP:
    def __init__(self, hidn_lay_num, hidn_node_num, ouput_num):
        self.hidn_lay_num = hidn_lay_num
        self.hidn_node_num = hidn_node_num
        self.output_num = ouput_num
        self.weights = []
        self.bias = []

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

        for i in range(1, len(self.weights)):
            dot_mat = Matrix.dot(z, self.weights[i].mat_t(return_mat=True))
            z = add_bias(dot_mat, self.bias[i])

        print(z.shape())
        return z
          
