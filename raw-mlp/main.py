from Matrix import Matrix
import random
from MLP import MLP

X = Matrix(10, 2)

X.matrix = [[1, 2], [3, 4], [5, 255], [9, 4], [2, 5], [1, 2], [3, 4], [5, 255], [9, 4], [2, 5]]
labels = Matrix(10, 1)

for i in range (labels.n_rows):
    labels.matrix[i][0] = random.randint(0, 9)

mlp = MLP(2, 2, 2, 5)

cost = mlp.forward(X, labels)
mlp.backwards_propagation(cost, labels)
