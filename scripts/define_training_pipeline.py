# steps for a pipeline:
# 1 Design the model (define number of inputs (input size) and outputs (output size, plus define forward pass with all different operations and layers)
# 2 Construct the loss and optimzer
# 3 Training loop
# -forward pass: compute prediction
# backward pass: gradients
# update weights
# iterate this until done

import torch
import torch.nn as nn

# f = w * x #function that is linear combination of weights and inputs (don't care about the bias here)

# f = 2 * x
X = torch.tensor([[1], [2], [3], [4]], dtype=torch.float32)
y = torch.tensor([[2], [4], [6], [8]], dtype=torch.float32)

X_test = torch.tensor([5], dtype=torch.float32)

n_samples, n_features = X.shape
print(n_samples, n_features)

#define model
input_size = n_features
output_size = n_features
#model= nn.Linear(input_size, output_size)

#if needed to define own model
class LinearRegression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()
        #define layers
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.lin(x)

model= LinearRegression(input_size, output_size)




print(f'Pred before training: f(5) = {model(X_test).item():.3f}')

# Training
learning_rate = 0.01
n_iters = 100

# loss
loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(n_iters):
    # prediction = forward pass
    y_pred = model(X)

    # loss
    l = loss(y, y_pred)

    # grads =backward pass
    l.backward()  # dl/dx (gradient of our loss wrt w)

    # update weights
    optimizer.step()

    # zero gradients
    optimizer.zero_grad()

    if epoch % 10 == 0:
        [w, b] = model.parameters()
        print(f"epoch {epoch + 1}: w = {w[0][0].item():.3f}, loss = {l:.8f}")

print(f'Pred after training: f(5) = {model(X_test).item():.3f}')
