import torch
import torch.nn as nn

'''
STEPS
# 1. Data,
# 2. DataLoader, Transformation,
# 3. Multilayer NN. ac func
# 4. loss and optimizer
# 5. training loop,
# 6. model eval,
# 7. GPU support
'''

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyperparams
input_size= 3272  #size of the 1d tensor
hidden_size= 100
num_classes= 6
num_epochs = 2
batch_size = 50
learning_rate = 0.001


#data


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no activation and no softmax at the end
        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (htmls, labels) in enumerate(train_loader):
        # origin shape: [100, 1, 28, 28]
        # resized: [100, 784]
        embeds = nn.Embedding(input_size, 5)
        htmls = embeds(htmls).to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(htmls)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{n_total_steps}], Loss: {loss.item():.4f}')

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for htmls, labels in test_loader:
        htmls = htmls.to(device)
        labels = labels.to(device)
        outputs = model(htmls)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on test examples: {acc} %')

