import torch

# Create a sample tensor
tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print("Original Tensor:\n", tensor)

# Accessing a single element
element = tensor[0, 1]  # Element at row 0, column 1
print("\nElement at (0, 1):", element)

# Accessing an entire row
row = tensor[1]  # Second row (index 1)
print("\nSecond Row:", row)

# Accessing an entire column
column = tensor[:, 2]  # All rows, column 2
print("\nThird Column:", column)

# Slicing (a subset of rows and columns)
slice_tensor = tensor[0:2, 1:3]  # Rows 0-1, Columns 1-2
print("\nSliced Tensor (Rows 0-1, Columns 1-2):\n", slice_tensor)

# Selecting specific rows or columns using indexing
rows_selected = tensor[[0, 2]]  # Select rows 0 and 2
print("\nRows 0 and 2:\n", rows_selected)

columns_selected = tensor[:, [0, 2]]  # Select columns 0 and 2
print("\nColumns 0 and 2:\n", columns_selected)

# Advanced: Boolean masking
mask = tensor > 5  # Create a mask for elements greater than 5
filtered_tensor = tensor[mask]  # Use the mask to filter elements
print("\nElements Greater than 5:", filtered_tensor)