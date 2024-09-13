import torch
import torch.nn as nn
import torch.optim as optim


# Updated model with root and branches
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.root = nn.Linear(10, 10)
        self.branch_1 = nn.Linear(10, 10)
        self.branch_2 = nn.Linear(10, 10)

    def forward(self, x):
        x = self.root(x)
        out1 = self.branch_1(x)
        out2 = self.branch_2(x)  # Detach to prevent gradients from flowing back to root
        return out1, out2


# Initialize the model and define the losses
model = SimpleModel()

# Define two losses for out1 and out2
criterion1 = nn.MSELoss()
criterion2 = nn.L1Loss()

# Sample input and target tensors
input_tensor = torch.randn(5, 10)
target_tensor1 = torch.randn(5, 10)
target_tensor2 = torch.randn(5, 10)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)


# Function to log weights and biases of layers
def log_weights_biases(model):
    root_weight, root_bias = (
        model.root.weight.clone().detach(),
        model.root.bias.clone().detach(),
    )
    branch_1_weight, branch_1_bias = (
        model.branch_1.weight.clone().detach(),
        model.branch_1.bias.clone().detach(),
    )
    branch_2_weight, branch_2_bias = (
        model.branch_2.weight.clone().detach(),
        model.branch_2.bias.clone().detach(),
    )
    return (
        (root_weight, root_bias),
        (branch_1_weight, branch_1_bias),
        (branch_2_weight, branch_2_bias),
    )


# Approach 1: Sum or Weighted Sum of Losses
def sum_of_losses_approach():
    # Forward pass
    out1, out2 = model(input_tensor)

    # Compute combined loss
    loss1 = criterion1(out1, target_tensor1)
    loss2 = criterion2(out2, target_tensor2)
    total_loss = loss1 + loss2

    # Backward pass and optimize
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return log_weights_biases(model)


# Approach 2: Sequential Loss Updates
def sequential_loss_updates_approach():
    # Loss 1
    out1, _ = model(input_tensor)
    loss1 = criterion1(out1, target_tensor1)
    optimizer.zero_grad()
    loss1.backward()
    optimizer.step()

    # Loss 2
    _, out2 = model(input_tensor)
    loss2 = criterion2(out2, target_tensor2)
    optimizer.zero_grad()
    loss2.backward()
    optimizer.step()

    return log_weights_biases(model)


# Run the comparison
# Reset the model weights to ensure a fair comparison
torch.manual_seed(0)
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Compare the two approaches
weights_sum = sum_of_losses_approach()

# Reset the model weights again for the second approach
torch.manual_seed(0)
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

weights_seq = sequential_loss_updates_approach()


# Print comparison results
def compare_tensors(tensor1, tensor2, name):
    if torch.equal(tensor1, tensor2):
        print(f"{name} are identical.")
    else:
        print(f"{name} differ.")


# Compare root, branch_1, and branch_2 weights and biases
compare_tensors(weights_sum[0][0], weights_seq[0][0], "root weights")
compare_tensors(weights_sum[0][1], weights_seq[0][1], "root biases")
compare_tensors(weights_sum[1][0], weights_seq[1][0], "branch_1 weights")
compare_tensors(weights_sum[1][1], weights_seq[1][1], "branch_1 biases")
compare_tensors(weights_sum[2][0], weights_seq[2][0], "branch_2 weights")
compare_tensors(weights_sum[2][1], weights_seq[2][1], "branch_2 biases")
