import torch

# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create a tensor with specific values, data type, and gradient requirement
my_tensor = torch.tensor(
    [[1, 2, 3], [4, 5, 6]],
    dtype=torch.float32,  # Specify the data type
    device=device,        # Assign it to the GPU or CPU
    requires_grad=True    # Enable gradient computation
)

# Print the tensor and its properties
print("Tensor:")
print(my_tensor)
print("Data Type:", my_tensor.dtype)
print("Device:", my_tensor.device)
print("Shape:", my_tensor.shape)
print("Requires Gradients:", my_tensor.requires_grad)

# Other common initialization methods for tensors
# 1. Create an empty tensor (uninitialized values)
empty_tensor = torch.empty(size=(3, 3))  # Uninitialized values
print("\nEmpty Tensor:")
print(empty_tensor)

# 2. Create a tensor filled with zeros
zeros_tensor = torch.zeros((3, 3))
print("\nZeros Tensor:")
print(zeros_tensor)

# 3. Create a tensor filled with random values
random_tensor = torch.rand((3, 3))  # Uniformly distributed values in [0, 1)
print("\nRandom Tensor:")
print(random_tensor)

# 4. Create a tensor filled with ones
ones_tensor = torch.ones((3, 3))
print("\nOnes Tensor:")
print(ones_tensor)

# 5. Create an identity matrix (square matrix with 1s on the diagonal)
identity_tensor = torch.eye(5, 5)
print("\nIdentity Matrix:")
print(identity_tensor)

# Additional Functions:

# 6. Create a tensor with a sequence of values (like range in Python)
arange_tensor = torch.arange(0, 10, step=2)  # Sequence from 0 to 10, step 2
print("\nArange Tensor:")
print(arange_tensor)

# 7. Create a tensor with evenly spaced values over a specified range
linspace_tensor = torch.linspace(0, 1, steps=5)  # 5 values from 0 to 1
print("\nLinspace Tensor:")
print(linspace_tensor)

# 8. Create a tensor with random values from a normal distribution
normal_tensor = torch.empty((3, 3)).normal_(mean=0, std=1)  # Mean 0, Std 1
print("\nNormal Distribution Tensor:")
print(normal_tensor)

# 9. Create a tensor with random values from a uniform distribution
uniform_tensor = torch.empty((3, 3)).uniform_(0, 1)  # Values in [0, 1)
print("\nUniform Distribution Tensor:")
print(uniform_tensor)

# 10. Create a diagonal matrix from a tensor
diag_tensor = torch.diag(torch.tensor([1, 2, 3]))
print("\nDiagonal Matrix:")
print(diag_tensor)


# Original tensor
tensor = torch.tensor([1.5, 2.3, 3.7])
print("Original Tensor:", tensor)
print("Original Data Type:", tensor.dtype)

# Convert to float
float_tensor = tensor.float()
print("\nConverted to Float:", float_tensor)

# Convert to int (truncates values)
int_tensor = tensor.int()
print("\nConverted to Int:", int_tensor)

# Convert to double (64-bit float)
double_tensor = tensor.double()
print("\nConverted to Double:", double_tensor)