# imports
import sklearn.metrics
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.table_data_loaders import NeuralNet
import torch
import torch.nn as nn
from transformers import PreTrainedTokenizerFast
import matplotlib.pyplot as plt
import matplotlib
from tableclass_engine_FFNN import train, validate, label_wise_metrics
import numpy as np

matplotlib.style.use('ggplot')

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

# ============ Set a value =============== #
torch.manual_seed(48)

# ============ Load and Check Tokenizer =========== #
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# ============ Get data loaders and datasets =============== #

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(
    inp_data_dir="../data/train-test-val",
    inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
    max_len=500, batch_size=50, val_batch_size=100,
    n_workers=0)

# ============ Set Config =============== #

# device config
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cpu'

input_size = 500  # size of the 1d tensor
num_classes = 10
hidden_size = 100
epochs = 10
batch_size = 50
lr = 0.001
embeds_size = 100
vocab_size = tokenizer.vocab_size + len(tokenizer.all_special_tokens)
padding_idx = tokenizer.pad_token_id

torch.autograd.set_detect_anomaly(True)

# ============ Get Model =============== #
model = NeuralNet(input_size=input_size, num_classes=num_classes, embeds_size=embeds_size, vocab_size=vocab_size,
                  padding_idx=padding_idx, hidden_size=hidden_size)  # .to(device)

# ============ Define Loss and Optimiser =============== #
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# ============ Training and Val Loop  =============== #
train_loss = []
valid_loss = []
for epoch in range(epochs):
    print(f"Epoch {epoch + 1} of {epochs}")
    train_epoch_loss = train(
        model, train_dataloader, optimizer, criterion, train_dataset, device
    )
    valid_epoch_loss = validate(
        model, valid_dataloader, criterion, valid_dataset, device
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f'Val Loss: {valid_epoch_loss:.4f}')

# ============ Save Model  =============== #
torch.save({
    'epoch': epochs,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': criterion,
}, '../data/outputs/model.pth')
# plot and save the train and validation line graphs
plt.figure(figsize=(10, 7))
plt.plot(train_loss, color='orange', label='train loss')
plt.plot(valid_loss, color='red', label='validataion loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('../data/outputs/loss.png')
plt.show()

# ============ Test Loop  =============== #

# load the model checkpoint
# checkpoint = torch.load('../data/outputs/model.pth')
# load model weights state_dict
# model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

with torch.no_grad():
    overall_correct = 0.
    total_predictions = 0.
    for counter, batch in enumerate(test_dataloader):
        input_ids = batch['input_ids'].to(device)
        target = batch['labels']
        outputs = model(input_ids)
        outputs = torch.sigmoid(outputs).cpu()
        # outputs = outputs.detach().cpu()
        predicted = np.round(outputs)

        total_predictions += target.size(0) * target.size(1)
        overall_correct += (predicted == target).sum().item()
        overall_accuracy = overall_correct / total_predictions * 100

        global_metrics, perclass_metrics = label_wise_metrics(target, predicted)

        labels = np.asarray(target)
        preds = np.asarray(predicted)
        class_report = sklearn.metrics.classification_report(labels, preds)

print(f"Overall Micro-Accuracy: {overall_accuracy}")
print(f"Classification Report: {class_report}")
print(f"Global Metrics: {global_metrics}")
print(f"Labelwise Metrics: {perclass_metrics}")
a = 1
