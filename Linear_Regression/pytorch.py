import torch
import torch.nn as nn
import torch.optim as optim



# X shape: (m, n) -> 5 samples, 1 feature
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]])

class LinearRegressionPyTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1) # 1 Feature -> 1 Output

    def forward(self,x):
        return self.linear(x)
    
model=LinearRegressionPyTorch() # Model Creation

# Define the loss function and optimizer:

creterion=nn.MSELoss() # Mean Squared Loss
optimizer=optim.SGD(model.parameters(),lr=0.01)

epochs=2000

### Trianing loop:

for epoch in range(epochs):
    y_pred=model.forward(X)
    loss=creterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()  # Calculating and optimizing the learning weights and Bias

with torch.no_grad():
    predictions=model(X)

print("Model Predictions:", predictions.squeeze().numpy())

# Extract learned parameters
w = model.linear.weight.item()
b = model.linear.bias.item()

print("Model parameters:")
print("Weight:", w)
print("Bias:", b)
