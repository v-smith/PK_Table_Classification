from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.table_data_loaders import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm

# ============ Set a value =============== #
torch.manual_seed(48)

# ============ Load and Check Tokenizer =========== #
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ============ Get data loaders =============== #

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(inp_data_dir="../data/train-test-val/",
                                                                       inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
                                                                       max_len=500, batch_size=50, val_batch_size=100,
                                                                       n_workers=0)


# ============ Set Config =============== #

# device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

input_size = 500  # size of the 1d tensor
hidden_size = 100
num_classes = 10
num_epochs = 10
batch_size = 50
learning_rate = 0.001
embeds_size = 110
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
padding_idx = tokenizer.pad_token_id

torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #
model = NeuralNet(input_size=input_size, num_classes=num_classes, embeds_size=embeds_size, vocab_size=vocab_size,
                  padding_idx=padding_idx, hidden_size=hidden_size) #.to(device)


# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# train the model
model.train()

# ============ Define Training Loop =============== #
for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1} of {num_epochs}")
    epoch_loss = []
    for batch in train_dataloader:
        input_ids = batch["input_ids"] #.to(device)
        labels = batch["labels"] #.to(device)

        logits = model(x=input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        epoch_loss.append(loss.item())

        # Backward and optimize
        optimizer.step()
        optimizer.zero_grad()

    print(f"Average Loss per Epoch: {(sum(epoch_loss) / len(epoch_loss)):.4f}")

# ============ Define Val Loop =============== #
# pass logits through sigmoid.... on validation set


# ============ Define Test Loop =============== #

with torch.no_grad():
    correct = 0.
    total = 0.
    #work out how to initiate loop
    for counter, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        target = batch['labels']
        outputs = model(input_ids)
        outputs = torch.sigmoid(outputs).cpu()
        #outputs = outputs.detach().cpu()
        predicted = np.round(outputs)
        total += target.size(0)  #try to understand this!!!
        correct += (predicted == target).sum().item()
accuracy = correct / total * 100
print("Overall Accuracy: {}%".format(accuracy))
