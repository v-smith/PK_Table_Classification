import torch
from torch import nn


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, embeds_size, vocab_size, padding_idx):
        super(NeuralNet, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.l1 = nn.Linear(embeds_size, hidden_size)  # or number of classes if only one layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.4)  # define proportion of neurons to drop out
        # self.double() #if feeding in a Torch.Double type

    def forward(self, x):
        embeddings = self.embedding(x)
        max_pooled = torch.max(embeddings, dim=1, keepdim=False)[0]
        #out_dropout = self.dropout(max_pooled)
        out_l1 = self.l1(max_pooled)
        #out_dropout = self.dropout(out_l1)
        out_relu = self.relu(out_l1)
        out_l2 = self.l2(out_relu)
        # no activation and no softmax at the end
        return out_l2
