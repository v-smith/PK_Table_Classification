import torch
print(torch.rand(5, 3))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device available now:', device)

