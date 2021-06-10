import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tokenizers import Tokenizer
import pandas as pd

'''
NOTES:
epoch = 1 forward and backward pass of ALL training samples 

batch_size = number of training samples in one forward and backward pass 

number of iterations = number of passes, each pass using [batch_size] number of samples 
e.g. 100 samples, batch_size= 20 --> 100/20 = 5 iterations for 1 epoch 
'''

class PKTestDataset():

    def __init__(self, html, targets):
        self.tokenizer = Tokenizer.from_file("../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
        self.html = torch.tensor(self.tokenizer.encode(html).ids) #padding #truncation??
        self.targets = torch.from_numpy(targets) # size [n_samples, n_labels]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.html[index], self.targets[index]

    def __len__(self):
        return len(self.html)


labels_dataset = pd.read_csv('../data/final-test-covs100.csv', sep=',')
y= labels_dataset.iloc[:, 2:].to_numpy()

test_dataset = PKTestDataset("../data/final-test-covs100html.txt", y, )

first_data = test_dataset[0]
features, labels = first_data
print(features, labels)


