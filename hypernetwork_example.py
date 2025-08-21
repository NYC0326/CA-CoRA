3
import torch
from hypernetwork import HyperNetwork, HyperLinear

# Define the dimensions
input_dim_hyper = 10  # Input dimension for the hypernetwork
output_dim_hyper = 256 * 128 + 128  # Output dimension for the hypernetwork (weights + bias)
hidden_dim_hyper = 32   # Hidden dimension for the hypernetwork

input_features_linear = 256  # Input features for the target layer
output_features_linear = 128 # Output features for the target layer

# Create a hypernetwork
hypernet = HyperNetwork(input_dim_hyper, output_dim_hyper, hidden_dim_hyper)

# Create a linear layer that uses the hypernetwork
hyper_linear_layer = HyperLinear(hypernet, input_features_linear, output_features_linear)

# Create some dummy data
batch_size = 4
input_to_hypernet = torch.randn(batch_size, input_dim_hyper)
input_to_linear_layer = torch.randn(batch_size, input_features_linear)

# Get the output from the hyper-linear layer
output = hyper_linear_layer(input_to_linear_layer, input_to_hypernet)

# Print the output shape
print("Output shape:", output.shape)
print("Output:", output)