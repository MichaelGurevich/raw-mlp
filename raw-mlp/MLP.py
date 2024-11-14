from Matrix import Matrix
import math
import random
import numpy as np

OUTPUT_SIZE = 10


def sigmoid(z):
    return 1 / (1 + math.e ** (-z))

def mini_batch_generator(X, y, n_batch, batch_size):
    combined = list(zip(X.matrix, y.matrix))
    random.shuffle(combined)
    X_shuffled, y_shuffled = zip(*combined)
    X_shuffled = list(X_shuffled)
    y_shuffled = list(y_shuffled)

    for i in range(n_batch):
        index = 0
        X_mini = Matrix(batch_size, X.cols)
        y_mini = Matrix(batch_size, 1)
        X_mini.matrix = X_shuffled[index: index+batch_size]
        y_mini.matrix = y_shuffled[index: index+batch_size]
        index += batch_size
        yield X_mini, y_mini
    



class MLP:
    def __init__(self, input_size, hidden_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = OUTPUT_SIZE
        self.w_hidden = Matrix(hidden_size, input_size)
        self.b_hidden = Matrix(hidden_size, 1)

        self.w_out = Matrix(OUTPUT_SIZE, hidden_size)
        self.b_out = Matrix(OUTPUT_SIZE, 1)

        rng = np.random.RandomState(123)

        self.w_hidden.matrix = rng.normal(
            loc=0.0, scale=0.1, size=(hidden_size, input_size)).tolist()
        
        self.w_out.matrix = rng.normal(
            loc=0.0, scale=0.1, size=(OUTPUT_SIZE, hidden_size)).tolist()

        
    def forward(self, X:Matrix):
        data_dot_w_h = Matrix.multiply(X, Matrix.transpose(self.w_hidden))
        z_h = Matrix.add_vector(data_dot_w_h, Matrix.transpose(self.b_hidden), axis=0)
        a_h = Matrix.apply_function_return(z_h, sigmoid)

        a_h_dot_w_out = Matrix.multiply(a_h, Matrix.transpose(self.w_out))
        z_o = Matrix.add_vector(a_h_dot_w_out, Matrix.transpose(self.b_out), axis=0)
        a_o = Matrix.apply_function_return(z_o, sigmoid)

        return a_h, a_o


    def y_to_one_hot(y:Matrix):
        # convert to one hot
        y_onehot = Matrix(y.rows, OUTPUT_SIZE)
        y_onehot.zeros()
        for i in range(y_onehot.rows):
            y_onehot.set_element(i, y.get_element(i, 0), 1)

        return y_onehot


    def cost(self, a_o:Matrix, y:Matrix):
        
        y_onehot = MLP.y_to_one_hot(y)

        total_cost = 0 
        for row in range(a_o.rows):
            for col in range(a_o.cols):
                error = y_onehot.get_element(row, col) - a_o.get_element(row, col)
                total_cost += error ** 2 
                        
            
        return total_cost / (a_o.rows * a_o.cols)
    
    
    
    def backwards(self, a_h, a_o, X, y):

        y_onehot = MLP.y_to_one_hot(y)

        # Part 1: update weights and bias for output layer
        d_L__d_a_o = Matrix(a_o.rows, a_o.cols)
        for row in range(d_L__d_a_o.rows):
            for col in range(d_L__d_a_o.cols):
                derivative = (2 * (a_o.get_element(row, col) -  y_onehot.get_element(row, col))) / a_o.rows 
                d_L__d_a_o.set_element(row, col, derivative)

        d_a_o__d_z_o = Matrix.apply_function_return(a_o, lambda a : a * (1 - a))
        d_z_o__d_w_o = a_h

        delta_o = Matrix.elementwise_multiply(d_L__d_a_o, d_a_o__d_z_o)

        d_L__d_w_o = Matrix.multiply(Matrix.transpose(delta_o), d_z_o__d_w_o)
        d_L__d_b_o = Matrix.transpose(Matrix.add_along_axis(delta_o, axis=0))
        
        # Part 2: update weights and bias for hidden layer

        d_z_o__d_a_h = self.w_out
        d_a_h__d_z_h = Matrix.apply_function_return(a_h, lambda a : a * (1 - a))
        d_z_h__d_w_h = X

        d_L__a_h = Matrix.multiply(delta_o, d_z_o__d_a_h)

        delta_h = Matrix.elementwise_multiply(d_L__a_h, d_a_h__d_z_h)
        d_L__d_w_h = Matrix.multiply(Matrix.transpose(delta_h), d_z_h__d_w_h)
        d_L__d_b_h = Matrix.transpose(Matrix.add_along_axis(delta_h, axis=0))
        
        return d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o
    

    def update_weight_bias(self, d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o, l_rate):
        updates = [d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o]
        for i in range(len(updates)):
            updates[i].multiply_by_scalar(l_rate)

        self.w_hidden = Matrix.subtract(self.w_hidden, updates[0])
        self.b_hidden = Matrix.subtract(self.b_hidden, updates[1])
        self.w_out = Matrix.subtract(self.w_out, updates[2])
        self.b_out = Matrix.subtract(self.b_out, updates[3])

    def fit(self,X, y, e=20,l_rate=0.1):
        for i in range(e):
            a_h, a_o = self.forward(X)
            print(f"Epoch: {i}, Cost: {self.cost(a_o, y)}")
            d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o = self.backwards(a_h, a_o, X, y)
            self.update_weight_bias(d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o, l_rate)

    def train(self, X, y, epochs=20, l_rate=0.1):

        for e in range(epochs):
            mini_batches = mini_batch_generator(X, y, 300, 100)
            mse = 0
            for X_mini, y_mini in mini_batches:
                a_h, a_o = self.forward(X_mini)
                mse += self.cost(a_o, y_mini)
                d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o = self.backwards(a_h, a_o, X_mini, y_mini)
                self.update_weight_bias(d_L__d_w_h, d_L__d_b_h, d_L__d_w_o, d_L__d_b_o, l_rate)

            mse /= 300
            print(f"Epoch: {e}, Cost: {mse}")

    def guess(self,X):
        a_h, a_o = self.forward(X)
        return a_o

X = Matrix(50000, 10)
y = Matrix(50000, 1)

for i in range(X.rows):
    random_num = random.randint(0,9)
    X.set_element(i, random_num, 1)
    y.set_element(i, 0, random_num)


exmp = Matrix(100, 10)
y_exmp = Matrix(100, 1)

for i in range(exmp.rows):
    random_num = random.randint(0,9)
    exmp.set_element(i, random_num, 1)
    y_exmp.set_element(i, 0, random_num)

model = MLP(10, 50)
#model.fit(X, y, 15, 0.01)



model.train(X, y, 12, l_rate=0.1)


a_o = model.guess(exmp)

wrong = 0
for row in range(a_o.rows):
    if a_o.matrix[row].index(max(a_o.matrix[row])) != y_exmp.matrix[row][0]:
        wrong += 1

print(f"right: {a_o.rows - wrong}, wrong: {wrong}")


