# imports
import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("../data/runs/")

matplotlib.style.use('ggplot')


# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print("====BEGIN TRAINING====")
    model.train()

    all_labels = []
    all_predictions = []
    running_loss = []
    counter= 0
    optimizer.zero_grad()
    for i, mini_batch in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1
        input_ids, target = mini_batch['input_ids'].to(device), mini_batch['labels'].to(device)
        # weight_rebal = torch.ones_like(target) / 95.0 + (1.0 - 1.0 / 95.0) * target #added, confirm this

        logits = model(input_ids)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(logits)
        loss = criterion(outputs, target)
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()  # updating weights
        optimizer.zero_grad()  # empties from memory

        assert outputs.shape == target.shape
        predicted = torch.round(outputs).detach().numpy()
        labels = target.detach().numpy()

        all_labels.extend(labels)
        all_predictions.extend(predicted)
        running_loss.append(loss.item())

    all_labels = np.stack(all_labels, axis=0)
    all_predictions = np.stack(all_predictions, axis=0)
    return all_labels, all_predictions, running_loss

'''
macro_f1 = f1_score(labels, predicted, average="macro")
class_report= classification_report(labels, predicted)
# loss = criterion(outputs, target, weight=weight_rebal) #added

if macro_f1 != 0.0:
    train_running_f1.append(macro_f1)
else:
    train_running_f1.append(None)
train_running_loss.append(loss.item())
'''



# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    running_loss= []
    counter= 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for i, mini_batch in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            input_ids, target = mini_batch['input_ids'].to(device), mini_batch['labels'].to(device)
            # weight_rebal = torch.ones_like(target) / 95.0 + (1.0 - 1.0 / 95.0) * target

            logits = model(input_ids)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(logits)
            loss = criterion(outputs, target)

            # loss = criterion(outputs, target, weight=weight_rebal)
            assert outputs.shape == target.shape
            predicted = torch.round(outputs).detach().numpy()
            labels = target.detach().numpy()
            all_predictions.extend(predicted)
            all_labels.extend(labels)
            running_loss.append(loss.item())


        all_labels = np.stack(all_labels, axis=0)
        all_predictions = np.stack(all_predictions, axis=0)
        return all_labels, all_predictions, running_loss

def f1_nozeros(class_report_dict):
    f1_list= []
    for d in class_report_dict.values():
        f1= d['f1-score']
        if f1 != 0:
            f1_list.append(f1)

    f1_nozeros= round(sum(f1_list)/len(f1_list),4)
    return f1_nozeros

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim=1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc


labels1 = np.array([[1, 0, 1, 0, 1],
                    [0, 1, 0, 1, 0],
                    [0, 0, 0, 0, 0],
                    [0, 1, 1, 0, 0]])

preds1 = np.array([[1, 1, 1, 1, 1],
                   [0, 0, 0, 1, 0],
                   [0, 0, 1, 0, 0],
                   [0, 1, 0, 0, 0]])


def label_wise_metrics(labels, preds):
    labels = np.asarray(labels)
    preds = np.asarray(preds)

    perclass_metrics = []
    all_correct_preds = []
    acc_list = []
    counter = 0
    for col, col2 in zip(labels.T, preds.T):
        correct_predictions = ((col == col2).sum())
        all_correct_preds.append(correct_predictions)
        class_f1 = round(f1_score(y_true=col, y_pred=col2), 4)
        class_acc = round(accuracy_score(y_true=col, y_pred=col2), 4)
        acc_list.append(class_acc)
        counter += 1
        metrics_dict = {"Label": counter,
                        "acc": class_acc,
                        "f1": class_f1}
        perclass_metrics.append(metrics_dict)

    micro_acc = round((sum(all_correct_preds) / (preds.size)), 4)
    macro_acc = round(sum(acc_list) / len(acc_list), 4)


    global_metrics = {"micro_acc": micro_acc,
                      "macro_acc": macro_acc}

    return global_metrics


def plot_loss_graph(train_loss, valid_loss, cf):
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color='orange', label='train loss')
    plt.plot(valid_loss, color='red', label='validataion loss')
    plt.title("Loss on Train and Validation Sets")
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(('../data/outputs/model_plots/loss-' + cf["run_name"] + '.png'))
    plt.show()


def plot_f1_graph(train_f1, valid_f1, cf):
    plt.figure(figsize=(10, 7))
    plt.plot(train_f1, color='orange', label='train f1')
    plt.plot(valid_f1, color='red', label='validataion f1')
    plt.title("Macro F1 Scores on Train and Validation sets")
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(('../data/outputs/model_plots/f1-' + cf["run_name"] + '.png'))
    plt.show()


'''
total_predictions += labels.size(0) * labels.size(1)
overall_correct += (predicted == target).sum().item()
label1_targets = np.asarray(target)[:, 1]
label1_preds = np.asarray(predicted)[:, 1]
label1_accuracy = label1_correct / target.size(0)

    return {'micro/precision': precision_score(y_true=target, y_pred=pred, average='micro'),
            'micro/recall': recall_score(y_true=target, y_pred=pred, average='micro'),
            'macro/precision': precision_score(y_true=target, y_pred=pred, average='macro'),
            'macro/recall': recall_score(y_true=target, y_pred=pred, average='macro'),
            'samples/precision': precision_score(y_true=target, y_pred=pred, average='samples'),
            'samples/recall': recall_score(y_true=target, y_pred=pred, average='samples'),
            'samples/f1': f1_score(y_true=target, y_pred=pred, average='samples'), }
'''
