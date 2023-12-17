import numpy as np

def NormeEuclidienne(vec):
    return np.linalg.norm(vec)


def transpose_vector(vector):
    return np.transpose(vector)

def power_iteration(A, tolerance, nombre_max_iterations):
    n = len(A)
    vecteur_initial = np.random.rand(n)
    lambda_prev = 0
    
    for iteration in range(1, nombre_max_iterations + 1):
        vecteur_propre_estime = np.dot(A, vecteur_initial)
        lambda_estime = np.dot(vecteur_initial, vecteur_propre_estime)
        vecteur_initial = vecteur_propre_estime / NormeEuclidienne(vecteur_propre_estime)
        
        if abs(lambda_estime - lambda_prev) < tolerance:
            return lambda_estime, vecteur_initial
        lambda_prev = lambda_estime
    return lambda_estime, vecteur_initial

def calculate_eigenvalues_and_vectors(A, tolerance, nombre_max_iterations):
    n = len(A)
    eigenvalues = np.zeros(n)
    eigenvectors = np.zeros((n, n))

    for i in range(n):
        eigenvalue, eigenvector = power_iteration(A, tolerance, nombre_max_iterations)
        eigenvalues[i] = eigenvalue
        eigenvectors[:, i] = eigenvector
        A = A - eigenvalue * np.outer(eigenvector, eigenvector.T)
    return eigenvalues, eigenvectors

def deflation(A, eigenvalue, eigenvector, nombre_max_iterations):
    n = len(A)
    B = A - eigenvalue * (eigenvector * transpose_vector(eigenvector))
    lambda2, vecteur2 = power_iteration(B, tolerance, nombre_max_iterations)
    return lambda2, vecteur2




A = np.array([[1, 3, 3],
              [-2, 11, -2],
              [8, -7, 6]])

tolerance = 1e-6
max_iterations = 100

eigenvalues, eigenvectors = calculate_eigenvalues_and_vectors(A, tolerance, max_iterations)

for i in range(len(eigenvalues)):
    print(f"Eigenvalue {i + 1}: {eigenvalues[i]}")
    print(f"Eigenvector {i + 1}:\n{eigenvectors[:, i]}\n")