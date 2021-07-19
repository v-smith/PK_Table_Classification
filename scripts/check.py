import torch
import numpy as np

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

weights= np.array([1,2,3,4,5,6,7,8,9])
weights= np.tile(weights, (64,1))
print(weights)
weights = torch.from_numpy(weights)

a=1
