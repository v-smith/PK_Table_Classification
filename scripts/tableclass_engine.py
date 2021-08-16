# imports
import shutil

import torch
import torchvision
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score, classification_report
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Iterable, Tuple

writer = SummaryWriter("../data/runs/")
matplotlib.style.use('ggplot')


# training function
def train(model, dataloader, optimizer, criterion, train_data, device):
    print("====BEGIN TRAINING====")
    model.train()
    all_labels = []
    all_predictions = []
    running_loss = 0.0
    counter = 0
    optimizer.zero_grad()
    for i, mini_batch in tqdm(enumerate(dataloader), total=int(len(train_data) / dataloader.batch_size)):
        counter += 1

        input_ids = mini_batch['input_ids'].to(device)
        target = mini_batch['labels'].to(device)
        # multi_hots = mini_batch["multi_hot"].to(device)

        optimizer.zero_grad()  # empties from memory
        logits = model(input_ids)
        # logits = model(input_ids, multi_hots)
        # apply sigmoid activation to get all the outputs between 0 and 1
        outputs = torch.sigmoid(logits)
        loss = criterion(outputs, target)
        # backpropagation
        # loss.backward() #changed this
        loss.backward()
        # update optimizer parameters
        optimizer.step()  # updating weights
        optimizer.zero_grad()  # empties from memory

        assert outputs.shape == target.shape
        predicted = torch.round(outputs).detach().numpy()
        labels = target.detach().numpy()

        all_labels.extend(labels)
        all_predictions.extend(predicted)
        running_loss += loss.item()

    final_loss = running_loss / counter
    all_labels = np.stack(all_labels, axis=0)
    all_predictions = np.stack(all_predictions, axis=0)
    return all_labels, all_predictions, final_loss


# validation function
def validate(model, dataloader, criterion, val_data, device):
    print('Validating')
    model.eval()
    running_loss = 0.0
    counter = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for i, mini_batch in tqdm(enumerate(dataloader), total=int(len(val_data) / dataloader.batch_size)):
            counter += 1
            input_ids = mini_batch['input_ids'].to(device)
            target = mini_batch['labels'].to(device)
            # multi_hots = mini_batch["multi_hot"].to(device)
            # weight_rebal = torch.ones_like(target) / 95.0 + (1.0 - 1.0 / 95.0) * target

            logits = model(input_ids)
            # logits = model(input_ids, multi_hots)
            # apply sigmoid activation to get all the outputs between 0 and 1
            outputs = torch.sigmoid(logits)
            loss = criterion(outputs, target)

            # loss = criterion(outputs, target, weight=weight_rebal)
            assert outputs.shape == target.shape
            predicted = torch.round(outputs).detach().numpy()
            labels = target.detach().numpy()
            all_predictions.extend(predicted)
            all_labels.extend(labels)
            running_loss += loss.item()

        final_loss = running_loss / counter
        all_labels = np.stack(all_labels, axis=0)
        all_predictions = np.stack(all_predictions, axis=0)
        return all_labels, all_predictions, final_loss


# overfit to training data for debugging
def overfit_subset(model, sub_sample, optimizer, criterion, device):
    print("====OVERFITTING SANITY CHECK====")
    model.train()

    input_ids = sub_sample['input_ids'].to(device)
    target = sub_sample['labels'].to(device)
    multi_hots = sub_sample["multi_hot"].to(device)

    optimizer.zero_grad()  # empties from memory
    logits = model(input_ids, multi_hots)
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

    final_loss = loss.item()
    return labels, predicted, final_loss


def save_checkpoint(state, is_best, run_name, file_path="../data/outputs/model_saves/"):
    filename = (file_path + run_name + "checkpoint.pth.tar")
    torch.save(state, filename)
    if is_best:
        is_best_filename = (file_path + run_name + "model_best.pth.tar")
        shutil.copyfile(filename, is_best_filename)


def f1_nozeros(class_report_dict: Dict) -> Tuple[float, float]:
    """Calculates F1-score without classes where support is zero"""
    acceptable_keys = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    non_zero_support_f1s = [dict(
        class_name=key,
        f1=v["f1-score"],
        support=v["support"]) for key, v in class_report_dict.items() if
        key in set(acceptable_keys) and v["support"] != 0]

    # if remove_nonrel:
    # non_zero_support_f1s = [x for x in non_zero_support_f1s if x["class_name"] != "4"]

    # if only_notrel:
    # non_zero_support_f1s = [x for x in non_zero_support_f1s if x["class_name"] == "4"]

    macrof1 = 0.
    weighted_f1 = 0.
    if non_zero_support_f1s:
        n_tp = np.sum([x["support"] for x in non_zero_support_f1s])
        macrof1 = np.mean([x["f1"] for x in non_zero_support_f1s])
        weighted_f1 = np.sum([x["f1"] * (x["support"] / n_tp) for x in non_zero_support_f1s])

    return macrof1, weighted_f1


def label_wise_metrics(labels, preds):
    """Calculates Label-wise and global metrics for all classes"""
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


def plot_f1_graph(train_f1, valid_f1, cf, variation: str):
    plt.figure(figsize=(10, 7))
    plt.plot(train_f1, color='orange', label='train f1')
    plt.plot(valid_f1, color='red', label='validataion f1')
    plt.title(f"Macro F1 Scores on Train and Validation sets {variation}")
    plt.xlabel('Epochs')
    plt.ylabel('F1')
    plt.legend()
    plt.savefig(('../data/outputs/model_plots/f1-' + variation + cf["run_name"] + '.png'))
    plt.show()

def plot_SA_graph(hyperparameters, scores):
    plt.figure(figsize=(10, 7))
    for num in range(len(hyperparameters)):
        plt.plot(scores[num], label=str(hyperparameters[num]))
    plt.legend()
    plt.title("Hyperparameter Comparison")
    plt.xlabel('Epochs')
    plt.ylabel('Weighted F1 Score')
    # plt.savefig(('../data/outputs/model_plots/loss-' + cf["run_name"] + '.png'))
    plt.show()

def plot_val_curve(hyperparameters, best_scores):
    plt.figure(figsize=(10, 7))
    plt.plot(hyperparameters, best_scores)
    plt.legend()
    plt.title("Validation Curve")
    plt.xlabel('lrs')
    plt.ylabel('Weighted F1 Score')
    # plt.savefig(('../data/outputs/model_plots/loss-' + cf["run_name"] + '.png'))
    plt.show()
