
import numpy as np 
 
def NormeEuclidienne(vec):
    return np.linalg.norm(vec)

def is_diagonally_dominant(matrix): 
    diagonal = np.abs(matrix.diagonal()) 
    row_sums = np.sum(np.abs(matrix), axis=1) - diagonal 
    return np.all(diagonal > row_sums) 
 
def make_diagonally_dominant(matrix): 
    n = len(matrix) 
    for i in range(n): 
        row_sum = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i]) 
        if np.abs(matrix[i, i]) <= row_sum: 
            max_diagonal_index = np.argmax(np.abs(matrix[:, i])) 
            matrix[[i, max_diagonal_index]] = matrix[[max_diagonal_index, i]] 
 
    return matrix 
 
def Jacobi(A, b, tolerance, max_iterations):
    n = len(b)
    x = np.zeros(n)
    x_prime = np.zeros(n)
    iteration = 0

    while iteration < max_iterations:
        for i in range(n):
            somme = 0
            for j in range(n):
                if j != i:
                    somme += A[i][j] * x[j]
            x_prime[i] = (b[i] - somme) / A[i][i]

        if NormeEuclidienne(x_prime - x) < tolerance:
            return x_prime  

        x = x_prime.copy()
        iteration += 1

    return x_prime
 
def fill_matrix():
    try:
        rows = int(input("Enter the number of rows: "))
        cols = int(input("Enter the number of columns: "))
        matrix = np.zeros((rows, cols))

        print("Enter matrix elements row-wise:")
        for i in range(rows):
            for j in range(cols):
                matrix[i, j] = float(input(f"Enter element at position ({i + 1}, {j + 1}): "))

        return matrix

    except:
        print("Error")
        return None
 
def fill_vec():
    try:
        cols = int(input("Enter the number of columns: "))
        vec = np.zeros((cols))

        print("Enter vecteur elements:")
        for i in range(cols):
                vec[i] = float(input(f"Enter element at position ({i + 1}): "))
        return vec
    except:
        print("Error")
        return None
 
    
A = fill_matrix()  
a=is_diagonally_dominant(A)
if a == False:
    print("Matrix A after make it diagonally dominant.")
    A=make_diagonally_dominant(A)
    print(A)
else:
    print(A)    
b = fill_vec() 

tolerance = float(input("Enter the tolerance: ")) 
max_iterations = int(input("Enter the number of iterations: ")) 
 
solution = Jacobi(A, b, tolerance, max_iterations) 
print("Solution:", solution) 
 
