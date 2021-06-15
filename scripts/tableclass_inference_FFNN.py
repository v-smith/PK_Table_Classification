import models
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# initialize the computation device
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device= "cpu"

# ============ Get Model =============== #
model = NeuralNet(input_size=input_size, num_classes=6, embeds_size=embeds_size, vocab_size=vocab_size,
                  padding_idx=padding_idx)  # .to(device)

# load the model checkpoint
checkpoint = torch.load('../data/outputs/model.pth')
# load model weights state_dict
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
