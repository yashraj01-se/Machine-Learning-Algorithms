import torch
import torch.nn as nn
import torch.optim as optim

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[0.0], [0.0], [0.0], [1.0], [1.0]])  # must be float for BCE

class LogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(1,1)

    def forward(self,X):
        return self.linear(X)

model=LogisticRegression()
epochs=2000

## loss Function and Optimizer:
creterion=nn.BCEWithLogitsLoss()
optimizer=optim.SGD(model.parameters(),lr=0.01)

### Training loop:
for epoch in range(epochs):
    y_pred=model(X)
    loss=creterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

with torch.no_grad():
    logits=model(X)
    probs=torch.sigmoid(logits)
    preds = (probs >= 0.5).int()

print("Predicted Probabilities:", probs.squeeze().numpy())
print("Predicted Classes:", preds.squeeze().numpy())

# Model parameters
w = model.linear.weight.item()
b = model.linear.bias.item()

print("Model parameters:")
print("Weight:", w)
print("Bias:", b)

    
