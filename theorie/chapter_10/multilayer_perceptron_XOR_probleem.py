import numpy as np

### KAN OP EXAMEN GEVRAGD WORDEN MET SIMPEL VOORBEELD / GETALLEN###

def step(z):
    return np.where(z >= 0, 1, 0)

W1 = np.array([[1, 1],[1, 1]])
b1 = np.array([-3/2, -1/2])

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

Z1 = X @ W1 + b1

print("Laag 1:\n____________________\n")

A1 = step(Z1)

print(A1)
print("____________________\nLaag 2:\n____________________\n")

W2 = np.array([[-1], [1]])
b2 = np.array([-1/2])

Z2 = A1 @ W2 + b2

A2 = step(Z2)

print(A2)