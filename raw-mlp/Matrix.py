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
    
