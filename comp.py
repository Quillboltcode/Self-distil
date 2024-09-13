import torch
import torch.nn as nn
import torch.optim as optim
import timm
import copy


# Define the model using timm's ResNet-18
class TimmResNet18WithBranches(nn.Module):
    def __init__(self, num_classes=1000):
        super(TimmResNet18WithBranches, self).__init__()
        resnet18 = timm.create_model("resnet18", pretrained=True, features_only=True)

        # Extract the feature stages
        self.layer0 = resnet18.feature_info[0]["module"]
        self.layer1 = resnet18.feature_info[1]["module"]
        self.layer2 = resnet18.feature_info[2]["module"]
        self.layer3 = resnet18.feature_info[3]["module"]
        self.layer4 = resnet18.feature_info[4]["module"]

        # Fully connected layers for each branch
        self.branch1_fc = nn.Linear(64, num_classes)
        self.branch2_fc = nn.Linear(128, num_classes)
        self.branch3_fc = nn.Linear(256, num_classes)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.layer0(x)

        x1 = self.layer1(x)
        branch1_out = torch.flatten(x1, 1)
        branch1_out = self.branch1_fc(branch1_out)

        x2 = self.layer2(x1)
        branch2_out = torch.flatten(x2, 1)
        branch2_out = self.branch2_fc(branch2_out)

        x3 = self.layer3(x2)
        branch3_out = torch.flatten(x3, 1)
        branch3_out = self.branch3_fc(branch3_out)

        x4 = self.layer4(x3)
        x4 = torch.flatten(x4, 1)
        main_out = self.fc(x4)

        return branch1_out, branch2_out, branch3_out, main_out


# Initialize the model, loss function, and optimizer
num_classes = 10
model = TimmResNet18WithBranches(num_classes=num_classes)
criterion = nn.CrossEntropyLoss()

# Copy the model to initialize the second approach with identical weights
model_multistep = copy.deepcopy(model)

# Define optimizer for both models
optimizer_sum = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
optimizer_multistep = optim.SGD(model_multistep.parameters(), lr=0.001, momentum=0.9)

# Weights for losses
weights = [0.2, 0.3, 0.5, 1.0]

# Example input and target
input_tensor = torch.randn(1, 3, 224, 224)
target = torch.tensor([0])
# Forward pass
branch1_out, branch2_out, branch3_out, main_out = model(input_tensor)

# Compute individual losses
loss1 = criterion(branch1_out, target)
loss2 = criterion(branch2_out, target)
loss3 = criterion(branch3_out, target)
loss4 = criterion(main_out, target)

# Compute weighted sum of losses
total_loss = (
    weights[0] * loss1 + weights[1] * loss2 + weights[2] * loss3 + weights[3] * loss4
)

# Zero gradients
optimizer_sum.zero_grad()

# Backward pass for the total loss
total_loss.backward()

# Step the optimizer
optimizer_sum.step()
# Forward pass
branch1_out, branch2_out, branch3_out, main_out = model_multistep(input_tensor)

# Compute individual losses
loss1 = criterion(branch1_out, target)
loss2 = criterion(branch2_out, target)
loss3 = criterion(branch3_out, target)
loss4 = criterion(main_out, target)

# Step 1: Branch 1
optimizer_multistep.zero_grad()
loss1.backward()
optimizer_multistep.step()

# Step 2: Branch 2
optimizer_multistep.zero_grad()
loss2.backward()
optimizer_multistep.step()

# Step 3: Branch 3
optimizer_multistep.zero_grad()
loss3.backward()
optimizer_multistep.step()

# Step 4: Main output
optimizer_multistep.zero_grad()
loss4.backward()
optimizer_multistep.step()
# Forward pass again to see the results
branch1_out_sum, branch2_out_sum, branch3_out_sum, main_out_sum = model(input_tensor)
branch1_out_ms, branch2_out_ms, branch3_out_ms, main_out_ms = model_multistep(
    input_tensor
)

# Print out losses to compare
print("Weighted Sum Approach Losses:")
print("Branch 1 Loss:", criterion(branch1_out_sum, target).item())
print("Branch 2 Loss:", criterion(branch2_out_sum, target).item())
print("Branch 3 Loss:", criterion(branch3_out_sum, target).item())
print("Main Output Loss:", criterion(main_out_sum, target).item())

print("\nMultistep Approach Losses:")
print("Branch 1 Loss:", criterion(branch1_out_ms, target).item())
print("Branch 2 Loss:", criterion(branch2_out_ms, target).item())
print("Branch 3 Loss:", criterion(branch3_out_ms, target).item())
print("Main Output Loss:", criterion(main_out_ms, target).item())
