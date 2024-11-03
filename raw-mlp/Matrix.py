class Matrix:
    def __init__(self, n_rows, n_cols):
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.matrix = [[0 for i in range(n_cols)] for j in range(n_rows)]


    def add_scalar(self, scalar):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.matrix[i][j] += scalar


    def mult_by_scalar(self, scalar):
        for i in range(self.n_rows):
            for j in range(self.n_cols):
                self.matrix[i][j] *= scalar
    
    def dot(self, mat):
        if self.n_cols != mat.n_rows:
            raise Exception(f"cannot dot {self.n_rows} x {self.n_cols} * {mat.n_rows} x {mat.n_cols}" )
        else:
            dot_mat = Matrix(self.n_rows, mat.n_cols)
            
            for i in range(dot_mat.n_rows):
                for j in range(dot_mat.n_cols):
                    sum = 0
                    for k in range(self.n_cols):
                        sum += self.matrix[i][k] * mat.matrix[k][j]

                    dot_mat.matrix[i][j] = sum
        
        return dot_mat
    

    def dot(mat1, mat2):
        if mat1.n_cols != mat2.n_rows:
            raise Exception(f"cannot dot {mat1.n_rows} x {mat1.n_cols} * {mat2.n_rows} x {mat2.n_cols}" )
         
        else:
            dot_mat = Matrix(mat1.n_rows, mat2.n_cols)
            
            for i in range(dot_mat.n_rows):
                for j in range(dot_mat.n_cols):
                    sum = 0
                    for k in range(mat1.n_cols):
                        sum += mat1.matrix[i][k] * mat2.matrix[k][j]

                    dot_mat.matrix[i][j] = sum
        
        return dot_mat
    

    def mat_add(self, mat):
        if self.n_rows != mat.n_rows or self.n_cols != mat.n_cols:
            raise Exception("Matrices are not the same dimentions")
        else:
            for i in range(self.n_rows):
                for j in range(self.n_cols):
                    self.matrix[i][j] += mat.matrix[i][j]

    
    def mat_add(mat1, mat2):
        if mat1.n_rows != mat2.n_rows or mat1.n_cols != mat2.n_cols:
            raise Exception("Matrices are not the same dimentions")
        else:
            add_mat = Matrix(mat1.n_rows, mat2.n_cols)
            for i in range(mat1.n_rows):
                for j in range(mat1.n_cols):
                    add_mat[i][j] += mat1.matrix[i][j] + mat2.matrix[i][j]

        return add_mat
