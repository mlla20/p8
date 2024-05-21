import torch

# Define parameters
num_tensors = 1000000
tensor_length = 40
file_path = "tensors_gaus_40.pt"

# Generate tensors
tensors = torch.randn(num_tensors, tensor_length)

# Write tensors to file
torch.save(tensors, file_path)

print("Tensors have been exported to", file_path)