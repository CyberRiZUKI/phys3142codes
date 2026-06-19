import numpy as np

# (a) Dot Product of Two 1D Arrays
print("\n(a) DOT PRODUCT")

a = np.random.rand(5)
b = np.random.rand(5)

print("Array A:", np.round(a, 2))
print("Array B:", np.round(b, 2))
print("Dot Product: ", round(np.dot(a, b), 2))


# (b) Max Value of Each Row and Column
print("\n(b) ROW & COLUMN MAXIMUMS")

grid = np.random.randint(0, 11, size=(4, 4))

print("Array:\n", grid)
print("Max of each row:", np.max(grid, axis=1))
print("Max of each column:", np.max(grid, axis=0))


# (c) Element-wise & Matrix Products
print("\n(c) ELEMENT-WISE & MATRIX PRODUCTS")

A = np.random.randint(1, 11, size=(3, 3))
B = np.random.randint(1, 11, size=(3, 3))

print("Matrix A:\n", A)
print("Matrix B:\n", B)
print("Element-wise Product:\n", np.multiply(A, B))
print("Matrix Product:\n", np.matmul(A, B))


# (d) Mean, Median, and Standard Deviation
print("\n(d) STATISTICAL MEASURES")
arr = np.random.rand(20)

print("Array:\n", np.round(arr, 2))
print(f"Mean:{np.mean(arr):.4f}")
print(f"Median:{np.median(arr):.4f}")
print(f"Standard Deviation: {np.std(arr):.4f}")
