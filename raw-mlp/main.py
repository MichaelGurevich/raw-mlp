from Matrix import Matrix
import random
from MLP import MLP

X_rows = 1000
X_cols = 10


X = Matrix(X_rows, X_cols)
y = Matrix(X_rows, 1)

X.matrix = [[0 for i in range(X_cols)] for i in range(X_rows)]

for i in range(X.n_rows):
    rand_num = random.randint(0, 9)
    X.matrix[i][rand_num] = 1
    y.matrix[i][0] = rand_num


mlp = MLP(10, 1, 10, 10, l_rate=0.1)

mlp.fit(20,X, y)

exmp = Matrix(1, 10)

exmp.matrix[0] = [0 for _ in range(X_cols)]

exmp.matrix[0][3] = 1
output = mlp.guess(exmp).matrix[0]
print(output.index(max(output)))

