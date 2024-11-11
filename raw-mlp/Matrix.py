import random

class Matrix:
    def __init__(self, rows, cols, **kwargs ):
        self.rows = rows
        self.cols = cols
        if "data" in kwargs:
            self.matrix = kwargs["data"]
        else:
            self.zeros()


    def print_matrix(self):
        for i in range(self.rows):
            print(f"{self.matrix[i]}")

    def shape(self):
        print(f"{self.rows}x{self.cols}")

    def get_element(self, row, col):
        if row < 0 or col < 0 or row > self.rows-1 or col > self.cols-1:
            raise Exception("Indexs are out of bounds")
        return self.matrix[row][col]

    def set_element(self, row, col, new_val):
        if row < 0 or col < 0 or row > self.rows-1 or col > self.cols-1:
            raise Exception("Indexs are out of bounds")
        self.matrix[row][col] = new_val
    
    def zeros(self):
        self.matrix = self.matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
    
    def ones(self):
        self.matrix = self.matrix = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

    def random(self):
        self.matrix = self.matrix = [[random.uniform(0,1) for _ in range(self.cols)] for _ in range(self.rows)]

    def add(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_elemet[i][j] + mat2.get_element[i][j])
            
        return output_mat
    
    def subtract(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_elemet[i][j] - mat2.get_element[i][j])
            
        return output_mat

    def multiply_by_scalar(mat, scalar):
        output_mat = Matrix(mat.rows, mat.cols)

        for i in range(mat.rows):
            for j in range(mat.cols):
                output_mat.set_element(i, j) = mat.get_elemet(i, j) * scalar
        
        return output_mat

    def elementwise_multiply(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_elemet[i][j] * mat2.get_element[i][j])
            
        return output_mat

    def multiply(mat1, mat2):
        if mat1.n_cols != mat2.n_rows:
            raise ValueError(f"Cannot perform dot product: {mat1.n_rows} x {mat1.n_cols} * {mat2.n_rows} x {mat2.n_cols}")

        # Initialize the result matrix with zeros
        dot_mat = Matrix(mat1.n_rows, mat2.n_cols)

        # Perform matrix multiplication
        for i in range(dot_mat.n_rows):
            for j in range(dot_mat.n_cols):
                dot_sum = 0
                for k in range(mat1.n_cols):
                    dot_sum += mat1.matrix[i][k] * mat2.matrix[k][j]

                dot_mat.matrix[i][j] = dot_sum

        return dot_mat
    
    def transpose(mat):
        transposed_mat = Matrix(mat.cols, mat.rows)

        for i in range(transposed_mat.rows):
            for j in range(transposed_mat.cols):
                transposed_mat.set_element(i, j, mat.get_element(j,i))

        return transposed_mat
