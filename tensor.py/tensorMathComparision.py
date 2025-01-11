import torch

# Define two tensors
x = torch.tensor([1, 2, 3])
y = torch.tensor([4, 5, 6])

# --- Addition ---
z1 = torch.empty(3)
torch.add(x, y, out=z1)
print("Addition (Method 1, with out):", z1)

z2 = torch.add(x, y)
print("Addition (Method 2):", z2)

z3 = x + y
print("Addition (Method 3):", z3)

# --- Subtraction ---
z4 = x - y
print("Subtraction:", z4)

# --- Division ---
z5 = torch.true_divide(x, y)
print("Division (Element-wise):", z5)

# --- In-place Operations ---
t = torch.zeros(3)
t.add_(x)
print("In-place Addition (using add_):", t)
t += x
print("In-place Addition (using +=):", t)

# --- Exponential Operations ---
z1 = x.pow(2)
print("Exponential (using pow):", z1)

z2 = x ** 2
print("Exponential (using **):", z2)

# --- Matrix Multiplication ---
matrix1 = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
matrix2 = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

matrix_result1 = torch.mm(matrix1, matrix2)
print("\nMatrix Multiplication (using torch.mm):")
print(matrix_result1)

matrix_result2 = matrix1 @ matrix2
print("Matrix Multiplication (using @ operator):")
print(matrix_result2)

# --- Matrix Exponential ---
matrix_exponential = torch.matrix_exp(matrix1)
print("\nMatrix Exponential:")
print(matrix_exponential)

# --- Sort ---
sorted_x, indices = torch.sort(x)
print("\nSorted Tensor:", sorted_x)
print("Indices of Sorted Elements:", indices)

# --- Argmax and Argmin ---
argmax_x = torch.argmax(x)
print("\nArgmax:", argmax_x)

argmin_x = torch.argmin(x)
print("Argmin:", argmin_x)

# --- Sum, Max, and Min ---
sum_x = torch.sum(x)
print("\nSum:", sum_x)

max_x = torch.max(x)
print("Max:", max_x)

min_x = torch.min(x)
print("Min:", min_x)

# --- Clamp ---
clamped_x = torch.clamp(x, min=2, max=3)  # Clamp values to the range [2, 3]
print("\nClamped Tensor:", clamped_x)
