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
        self.matrix = [[0 for _ in range(self.cols)] for _ in range(self.rows)]
    
    def ones(self):
        self.matrix = [[1 for _ in range(self.cols)] for _ in range(self.rows)]

    def random(self):
        self.matrix = [[random.uniform(0,1) for _ in range(self.cols)] for _ in range(self.rows)]

    def add(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_element(i,j) + mat2.get_element(i,j))
            
        return output_mat
    
    def subtract(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_element(i,j) - mat2.get_element(i,j))
            
        return output_mat

    def multiply_by_scalar(self, scalar):
        for i in range(self.rows):
            for j in range(self.cols):
                self.set_element(i, j, self.get_element(i, j) * scalar)

    def elementwise_multiply(mat1, mat2):
        if mat1.rows != mat2.rows or mat1.cols != mat2.cols:
            raise Exception("Matrixes are not the same dimensions")

        output_mat = Matrix(mat1.rows, mat1.cols)
        for i in range(mat1.rows):
            for j in range(mat1.cols):
                output_mat.set_element(i, j, mat1.get_element(i, j) * mat2.get_element(i, j))
            
        return output_mat

    def multiply(mat1, mat2):
        if mat1.cols != mat2.rows:
            raise ValueError(f"Cannot perform dot product: {mat1.rows} x {mat1.cols} * {mat2.rows} x {mat2.cols}")

        # Initialize the result matrix with zeros
        dot_mat = Matrix(mat1.rows, mat2.cols)

        # Perform matrix multiplication
        for i in range(dot_mat.rows):
            for j in range(dot_mat.cols):
                dot_sum = 0
                for k in range(mat1.cols):
                    dot_sum += mat1.get_element(i,k) * mat2.get_element(k,j)

                dot_mat.set_element(i,j, dot_sum)

        return dot_mat
    
    def transpose(mat):
        transposed_mat = Matrix(mat.cols, mat.rows)

        for i in range(transposed_mat.rows):
            for j in range(transposed_mat.cols):
                transposed_mat.set_element(i, j, mat.get_element(j,i))

        return transposed_mat
    
    def apply_function(self, func):
        for i in range(self.rows):
            for j in range(self.cols):
                self.set_element(i, j, func(self.get_element(i, j)))
    
    def apply_function_return(mat, func):
        new_mat = Matrix(mat.rows, mat.cols)
        for i in range(mat.rows):
            for j in range(mat.cols):
                new_mat.set_element(i, j, func(mat.get_element(i, j)))
        return new_mat
    
    def add_vector(mat, vector, axis=0):
       
        # Check the dimensions based on the axis argument
        if axis == 0:  # Add a row vector to each row
            if mat.cols != vector.cols:
                raise ValueError(f"Cannot perform vector addition: {mat.rows} x {mat.cols} * {vector.rows} x {vector.cols}")
        elif axis == 1:  # Add a column vector to each column
            if mat.rows != vector.rows:
                raise ValueError(f"Cannot perform vector addition: {mat.rows} x {mat.cols} * {vector.rows} x {vector.cols}")
        else:
            raise ValueError("Invalid axis. Axis must be 0 (row-wise) or 1 (column-wise).")

        # Create a new matrix for the result
        new_mat = Matrix(mat.rows, mat.cols)

        if axis == 0:  # Row-wise addition
            for i in range(mat.rows):
                for j in range(mat.cols):
                    new_mat.set_element(i, j, mat.get_element(i, j) + vector.get_element(0, j))
        elif axis == 1:  # Column-wise addition
            for i in range(mat.rows):
                for j in range(mat.cols):
                    new_mat.set_element(i, j, mat.get_element(i, j) + vector.get_element(i, 0))

        return new_mat

    def add_along_axis(mat, axis=0):
        
        if axis == 0:
            output_mat = Matrix(1, mat.cols)
            for col in range(mat.cols):
                col_sum = 0
                for row in range(mat.rows):
                    col_sum += mat.get_element(row, col)
                output_mat.set_element(0, col, col_sum)
        elif axis == 1:
            output_mat = Matrix(mat.rows, 1)
            for row in range(mat.rows):
                row_sum = 0
                for col in range(mat.cols):
                    row_sum += mat.get_element(row, col)
                output_mat.set_element(row, 0, row_sum)

        return output_mat
