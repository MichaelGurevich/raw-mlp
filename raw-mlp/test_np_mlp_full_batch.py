
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def create_X_y(num_examples, num_features):

    # Initialize the features matrix with zeros
    X = np.zeros((num_examples, num_features), dtype=int)

    # Generate labels and populate the features matrix
    y = np.random.randint(0, num_features, size=num_examples)

    # Set the corresponding feature to 1 based on the label for each example
    for i, label in enumerate(y):
        X[i, label] = 1

    return X, y


X, y = create_X_y(200, 10)

##########################
### MODEL
##########################

def sigmoid(z):                                        
    return 1. / (1. + np.exp(-z))


def int_to_onehot(y, num_labels):

    ary = np.zeros((y.shape[0], num_labels))
    for i, val in enumerate(y):

        ary[i, val] = 1

    return ary


class NeuralNetMLP:

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()
        
        self.num_classes = num_classes
        
        # hidden
        rng = np.random.RandomState(random_seed)
        
        self.weight_h = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features))
        self.bias_h = np.zeros(num_hidden)
        
        # output
        self.weight_out = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_out = np.zeros(num_classes)
        
    def forward(self, x):
        # Hidden layer
        # input dim: [n_examples, n_features] dot [n_hidden, n_features].T
        # output dim: [n_examples, n_hidden]
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)

        # Output layer
        # input dim: [n_examples, n_hidden] dot [n_classes, n_hidden].T
        # output dim: [n_examples, n_classes]
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        return a_h, a_out

    def backward(self, x, a_h, a_out, y):  
    
        #########################
        ### Output layer weights
        #########################
        
        # onehot encoding
        y_onehot = int_to_onehot(y, self.num_classes)

        # Part 1: dLoss/dOutWeights
        ## = dLoss/dOutAct * dOutAct/dOutNet * dOutNet/dOutWeight
        ## where DeltaOut = dLoss/dOutAct * dOutAct/dOutNet
        ## for convenient re-use
        
        # input/output dim: [n_examples, n_classes]
        d_loss__d_a_out = 2.*(a_out - y_onehot) / y.shape[0]

        # input/output dim: [n_examples, n_classes]
        d_a_out__d_z_out = a_out * (1. - a_out) # sigmoid derivative

        # output dim: [n_examples, n_classes]
        delta_out = d_loss__d_a_out * d_a_out__d_z_out # "delta (rule) placeholder"

        # gradient for output weights
        
        # [n_examples, n_hidden]
        d_z_out__dw_out = a_h
        
        # input dim: [n_classes, n_examples] dot [n_examples, n_hidden]
        # output dim: [n_classes, n_hidden]
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)
        d_loss__db_out = np.sum(delta_out, axis=0)
        

        #################################        
        # Part 2: dLoss/dHiddenWeights
        ## = DeltaOut * dOutNet/dHiddenAct * dHiddenAct/dHiddenNet * dHiddenNet/dWeight
        
        # [n_classes, n_hidden]
        d_z_out__a_h = self.weight_out
        
        # output dim: [n_examples, n_hidden]
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)
        
        # [n_examples, n_hidden]
        d_a_h__d_z_h = a_h * (1. - a_h) # sigmoid derivative
        
        # [n_examples, n_features]
        d_z_h__d_w_h = x
        
        # output dim: [n_hidden, n_features]
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, d_z_h__d_w_h)
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return (d_loss__dw_out, d_loss__db_out, 
                d_loss__d_w_h, d_loss__d_b_h)




model = NeuralNetMLP(num_features=10,
                     num_hidden=50,
                     num_classes=10)


# ## Coding the neural network training loop

# Defining data loaders:




num_epochs = 50
minibatch_size = 100


def minibatch_generator(X, y, minibatch_size):
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    for start_idx in range(0, indices.shape[0] - minibatch_size 
                           + 1, minibatch_size):
        batch_idx = indices[start_idx:start_idx + minibatch_size]
        
        yield X[batch_idx], y[batch_idx]

        
def compute_mse_and_acc(nnet, X, y, num_labels=10, minibatch_size=100):
    mse, correct_pred, num_examples = 0., 0, 0
    
        
    

    _, probas = nnet.forward(X)
    predicted_labels = np.argmax(probas, axis=1)
        
    onehot_targets = int_to_onehot(y, num_labels=num_labels)
    loss = np.mean((onehot_targets - probas)**2)
    correct_pred += (predicted_labels == y).sum()
        
    num_examples += y.shape[0]
    mse += loss

    acc = correct_pred/num_examples
    return mse, acc


def train(model, X, y, num_epochs,
          learning_rate=0.1):
    
    epoch_loss = []
    epoch_train_acc = []
    epoch_valid_acc = []
    
    for e in range(num_epochs):

        # Generate a shuffled index
        shuffled_indices = np.random.permutation(X.shape[0])

        # Shuffle both matrices using the same indices
        shuffled_X = X[shuffled_indices]
        shuffled_y = y[shuffled_indices]

        #### Compute outputs ####
        a_h, a_out = model.forward(X)

        #### Compute gradients ####
        d_loss__d_w_out, d_loss__d_b_out, d_loss__d_w_h, d_loss__d_b_h = model.backward(shuffled_X, a_h, a_out, shuffled_y)

        #### Update weights ####
        model.weight_h -= learning_rate * d_loss__d_w_h
        model.bias_h -= learning_rate * d_loss__d_b_h
        model.weight_out -= learning_rate * d_loss__d_w_out
        model.bias_out -= learning_rate * d_loss__d_b_out
            
            
        
        #### Epoch Logging ####        
        train_mse, train_acc = compute_mse_and_acc(model, shuffled_X, shuffled_y)
        
        epoch_train_acc.append(train_acc)
        epoch_loss.append(train_mse)
        print(f'Epoch: {e+1:03d}/{num_epochs:03d} '
              f'| Train MSE: {train_mse:.2f} '
              f'| Train Acc: {train_acc:.2f}% ')

    return epoch_loss, epoch_train_acc, epoch_valid_acc



def guess(model:NeuralNetMLP, X, y):
    _, a_o = model.forward(X)
    max_indices = np.argmax(a_o, axis=1)
    wrong = sum(max_indices != y)
    print(f"Correct {X.shape[0] - wrong}, Wrong: {wrong}")




np.random.seed(123) # for the training set shuffling

epoch_loss, epoch_train_acc, epoch_valid_acc = train(
    model, X, y,
    num_epochs=20000, learning_rate=0.1)


X_test, y_test = create_X_y(100, 10)
guess(model, X_test, y_test)
