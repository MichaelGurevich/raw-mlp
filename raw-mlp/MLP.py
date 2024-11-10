
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
    def __init__(self,n_features, hidn_lay_num, hidn_node_num, ouput_num, l_rate=10):
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
                cost += ((self.a_[-1].matrix[i][j] - labels.matrix[i][j])**2 ) / X.n_rows

            cost_mat.matrix[j][0] = cost / z.n_rows
            
        return cost_mat
    
    def backwards(self, cost_mat: Matrix, X:Matrix, y:Matrix):
        

        ''' Part 1: calculate the gradient to update the weights and bias of the ouput layer'''
        labels = convert_labels(y)

        # [n_examples, n_ouput_nodes]
        d_L__d_a_out = Matrix(self.a_[-1].n_rows, self.a_[-1].n_cols)
        for i in range(self.a_[-1].n_rows):
            for j in range(self.a_[-1].n_cols):
                d_L__d_a_out.matrix[i][j] = (2 * (self.a_[-1].matrix[i][j] - labels.matrix[i][j])) / labels.n_rows
                
    
        # [n_examples, n_ouput_nodes]
        d_a_out__d_z_out = Matrix.element_oper_return(self.a_[-1], lambda a : a * (1 - a))

        # [n_examples, n_features]
        d_z_out__d_w_out = self.a_[-2]

    
        delta_output = Matrix(self.a_[-1].n_rows, self.a_[-1].n_cols)
    
        for k in range(self.w_[-1].n_cols):
            for i in range(self.w_[-1].n_rows):
                delta_output.matrix[i][k] = d_L__d_a_out.matrix[i][k] * d_a_out__d_z_out.matrix[i][k]

        d_L__d_w_out = Matrix.dot(delta_output.mat_t(return_mat=True), d_z_out__d_w_out)
        

        d_L__d_b_out = Matrix(self.w_[-1].n_rows, 1)

        for i in range(delta_output.n_cols):
            col_sum = 0
            for j in range(delta_output.n_rows):
                col_sum += delta_output.matrix[j][i]
            d_L__d_b_out.matrix[i][0] = col_sum

        '''Part 2: calculate the gradient to update the weights and bias of the hidden layer
        d_z_out__d_a_h
        d_a_h__d_z_h
        d_z_h__d_w_h
        '''

        #print(delta_output.shape()) # 20 x 10
        d_z_h__d_w_h = X # 20 x 10
        d_z_out__d_a_h = self.w_[-1] # 10 x 5
        d_a_h__d_z_h = Matrix.element_oper_return(self.a_[-2], lambda a : a * (1 - a)) # 20 x 5
    

        d_loss__a_h = Matrix.dot(delta_output, d_z_out__d_a_h) # 20 x 5
        
        delta_hidden = Matrix(d_loss__a_h.n_rows, d_loss__a_h.n_cols)
        for i in range(d_loss__a_h.n_rows):
            for j in range(d_loss__a_h.n_cols):
                delta_hidden.matrix[i][j] = d_loss__a_h.matrix[i][j] * d_a_h__d_z_h.matrix[i][j]
            
        d_L__d_w_h = Matrix.dot(delta_hidden.mat_t(return_mat=True), d_z_h__d_w_h)
        

        d_L__d_b_h = Matrix(self.w_[-2].n_rows, 1)

        for i in range(delta_hidden.n_cols):
            col_sum = 0
            for j in range(delta_hidden.n_rows):
                col_sum += delta_hidden.matrix[j][i]
            col_sum /= delta_hidden.n_rows
            d_L__d_b_h.matrix[i][0] = col_sum


        return (d_L__d_w_out, d_L__d_b_out, d_L__d_w_h, d_L__d_b_h)


        

    def fit(self,e:int, X:Matrix, y:Matrix):


        for _ in range(e):
            zip_list = list(zip(X.matrix, y.matrix))
            
            X_shuffled = Matrix(X.n_rows, X.n_cols)
            y_shuffled = Matrix(y.n_rows, y.n_cols)

            random.shuffle(zip_list)

            X_shuffled.matrix, y_shuffled.matrix = zip(*zip_list)

            X_shuffled.matrix = list(X_shuffled.matrix)
            y_shuffled.matrix = list(y_shuffled.matrix)
            
            cost = self.forward(X, y)
            #print(cost.matrix[:5])
            d_L__d_w_out, d_L__d_b_out, d_L__d_w_h, d_L__d_b_h = self.backwards(cost, X, y)
            d_L__d_w_out.mult_by_scalar(-self.l_rate)
            d_L__d_b_out.mult_by_scalar(-self.l_rate)
            d_L__d_w_h.mult_by_scalar(-self.l_rate)
            d_L__d_b_h.mult_by_scalar(-self.l_rate)

            self.w_[-1].mat_add(d_L__d_w_out)
            self.b_[-1].mat_add(d_L__d_b_out)
            self.w_[-2].mat_add(d_L__d_w_h)
            self.b_[-2].mat_add(d_L__d_b_h)
            
    def guess(self, x):
        self.a_ = []
        dot_mat = Matrix.dot(x, self.w_[0].mat_t(return_mat=True))
        z = add_bias(dot_mat, self.b_[0])
        z.element_oper(sigmoid)
        self.a_.append(z)

        for i in range(1, len(self.w_)):
            dot_mat = Matrix.dot(z, self.w_[i].mat_t(return_mat=True))
            z = add_bias(dot_mat, self.b_[i])
            z.element_oper(sigmoid)
            self.a_.append(z)

        return self.a_[-1]
        
      
        
        
       

          
