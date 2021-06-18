import torch
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import numpy as np
from typing import Dict, List, Iterable
from ignite.metrics import Accuracy

# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print('Training')
    model.train()
    counter = 0
    train_running_loss = 0.0
    for i, data in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1
        data, target = data['input_ids'].to(device), data['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(data)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(outputs)
        loss = criterion(outputs, target)
        train_running_loss += loss.item()
        # backpropagation
        loss.backward()
        # update optimizer parameters
        optimizer.step()

    train_loss = train_running_loss / counter
    return train_loss


# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    counter = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            data, target = data['input_ids'].to(device), data['labels'].to(device)
            outputs = model(data)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(outputs)
            loss = criterion(outputs, target)
            val_running_loss += loss.item()

        val_loss = val_running_loss / counter
        return val_loss


labels1= np.array([[1, 0, 1, 0, 1],
       [0, 1, 0, 1, 0],
       [0, 0, 0, 0, 0],
       [0, 1, 1, 0, 0]])

preds1= np.array([[1, 1, 1, 1, 1],
       [0, 0, 0, 1, 0],
       [0, 0, 1, 0, 0],
       [0, 1, 0, 0, 0]])


def label_wise_metrics(labels, preds):

    labels= np.asarray(labels)
    preds= np.asarray(preds)

    perclass_metrics = []
    all_correct_preds= []
    acc_list = []
    counter= 0
    for col, col2 in zip(labels.T,preds.T):
        correct_predictions = ((col  == col2).sum())
        all_correct_preds.append(correct_predictions)
        class_f1 = round(f1_score(y_true=col, y_pred=col2),4)
        class_acc = round(accuracy_score(y_true=col, y_pred=col2),4)
        acc_list.append(class_acc)
        counter+=1
        metrics_dict= {"Label": counter, "acc": class_acc, "f1": class_f1}
        perclass_metrics.append(metrics_dict)



    micro_acc =accuracy_score(labels, preds)
    macro_acc= sum(acc_list)/len(acc_list)
    micro_f1= round(f1_score(labels, preds, average="micro"), 4)
    macro_f1= round(f1_score(labels, preds, average="macro"),4)

    global_metrics= {"micro_acc": micro_acc, "macro_acc": macro_acc, "micro_f1": micro_f1, "macro_f1": macro_f1}

    print(global_metrics)
    print(perclass_metrics)



label_wise_metrics(labels1, preds1)

'''
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

