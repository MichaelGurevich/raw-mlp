from Matrix import Matrix
import random
from MLP import MLP

new_mat = Matrix(5, 2)

new_mat.matrix = [[1, 2], [3, 4], [5, 6], [9, 4], [2, 5]]


mlp = MLP(2, 15, 1)

print(mlp.feed_forward(new_mat).matrix)
