from Matrix import Matrix
import random
import math

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
    return one_hot_data


class MLP:
    def __init__(self,n_features, hidn_lay_num, hidn_node_num, ouput_num, l_rate=0.1):
        self.hidn_lay_num = hidn_lay_num
        self.hidn_node_num = hidn_node_num
        self.output_num = ouput_num
        self.l_rate = l_rate
        self.initialize_weights_bias(n_features)


    def init_dim_list(self, n_features):
        self.dim_list = []
        self.dim_list.append((self.hidn_node_num, n_features))
        for i in range(1, self.hidn_lay_num):
            self.dim_list.append((self.hidn_node_num, self.hidn_node_num))
        self.dim_list.append((self.output_num, self.hidn_node_num))

    
    def initialize_weights_bias(self, n_features):
        self.init_dim_list(n_features)

        self.w_ = []
        self.b_ = []

        for i in range(len(self.dim_list)):
            w = Matrix(self.dim_list[i][0], self.dim_list[i][1])
            b = Matrix(self.dim_list[i][0],1)
            w.element_oper(lambda x : random.uniform(-1, 1))
            b.element_oper(lambda x : random.uniform(-1, 1))
            self.w_.append(w)
            self.b_.append(b)

    def forward(self, X, y):
        self.a_ = []
        dot_mat = Matrix.dot(X, self.w_[0].mat_t(return_mat=True))
        z = add_bias(dot_mat, self.b_[0])
        z.element_oper(sigmoid)
        self.a_.append(z)

        for i in range(1, len(self.w_)):
            dot_mat = Matrix.dot(z, self.w_[i].mat_t(return_mat=True))
            z = add_bias(dot_mat, self.b_[i])
            z.element_oper(sigmoid)
            self.a_.append(z)

        labels = convert_labels(y)
        cost = 0
        cost_mat = Matrix(z.n_cols, 1)
        for j in range(z.n_cols):
            for i in range(z.n_rows):
                cost += (z.matrix[i][j] - labels.matrix[i][j])**2

            cost_mat.matrix[j][0] = cost / z.n_rows
            
        return cost_mat
    
    def backwards_propagation(self, cost_mat: Matrix, y):

        '''
        Part 1:
        - Get the gradient for the w_ and b_for the output layer 
        d_L__d_a 
        d_a__d_z
        d_z__d_w
        '''
        
        labels = convert_labels(y)
        d_L__d_a = Matrix(self.a_[-1].n_rows, self.a_[-1].n_cols)
        for i in range(self.a_[-1].n_rows):
            for j in range(self.a_[-1].n_cols):
                d_L__d_a.matrix[i][j] = 2 * (self.a_[-1].matrix[i][j] - labels.matrix[i][j])
                
        
        d_a__d_z = Matrix.element_oper_return(self.a_[-1], lambda a : a * (1 - a))


        # getting the gradient for updating the output w_
        
        
       

          
