from table_data_loaders import get_dataloaders
from table_data_loaders import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast

dataloaders_train, dataloaders_val, dataloaders_test = get_dataloaders(inp_data_dir="../data/",
                                                                       inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
                                                                       max_len=500, batch_size=50, val_batch_size=100,
                                                                       n_workers=0)

tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

input_size = 500  # size of the 1d tensor
# hidden_size = 100
num_classes = 6
num_epochs = 10
batch_size = 50
learning_rate = 0.001
embeds_size = 110
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
padding_idx = tokenizer.pad_token_id

torch.autograd.set_detect_anomaly(True)

model = NeuralNet(input_size=input_size, num_classes=6, embeds_size=embeds_size, vocab_size=vocab_size,
                  padding_idx=padding_idx)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
model.train()

for epoch in range(0, num_epochs):
    epoch_loss= []
    for batch in dataloaders_train:
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        logits = model(x=input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        epoch_loss.append(loss.item())
        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

    print(sum(epoch_loss)/len(epoch_loss))
