import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import jsonlines
from sklearn.preprocessing import MultiLabelBinarizer

#matplotlib.style.use('seaborn-white')
matplotlib.style.use('ggplot')

with jsonlines.open("../data/train-test-val/train-corrected.jsonl") as reader:
    train_list = []
    for obj in reader:
        train_list.append(obj)

with jsonlines.open("../data/train-test-val/val-corrected.jsonl") as reader:
    val_list = []
    for obj in reader:
        val_list.append(obj)

with jsonlines.open("../data/train-test-val/test-corrected.jsonl") as reader:
    test_list = []
    for obj in reader:
        test_list.append(obj)

train_df= pd.DataFrame(train_list)
train_labels= list(train_df["accept"])

val_df= pd.DataFrame(val_list)
val_labels= list(val_df["accept"])

test_df= pd.DataFrame(test_list)
test_labels= list(test_df["accept"])

mlb = MultiLabelBinarizer()
all_transformed_labels_train = mlb.fit_transform(train_labels)
all_transformed_labels_val = mlb.fit_transform(val_labels)
all_transformed_labels_test = mlb.fit_transform(test_labels)

all_transformed_df_train= pd.DataFrame(all_transformed_labels_train)
all_transformed_df_val= pd.DataFrame(all_transformed_labels_val)
all_transformed_df_test= pd.DataFrame(all_transformed_labels_test)

def get_class_totals(dummy_all, train:bool, val:bool, test:bool):
    labels = list(dummy_all.columns.values)
    counts = []
    for i in labels:
        counts.append((i, dummy_all[i].sum()))
    df_totals = pd.DataFrame(counts, columns=['label', 'number_of_tables'])
    if train:
        df_totals['label'] = df_totals["label"].map({0:0, 1:1, 2:2, 3:3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9})
        df_totals= df_totals.append({"label": 4, "number_of_tables": 325}, ignore_index=True)
    elif val:
        df_totals['label'] = df_totals["label"].map({0: 0, 1: 1, 2: 2, 3: 3, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9})
        df_totals = df_totals.append({"label": 4, "number_of_tables": 64}, ignore_index=True)
    elif test:
        df_totals= df_totals.drop(4, axis=0)
        df_totals= df_totals.append({"label": 4, "number_of_tables": 124}, ignore_index=True)
    return df_totals

train_label_totals= get_class_totals(all_transformed_df_train, train=True, val=False, test=False)
val_label_totals= get_class_totals(all_transformed_df_val, train=False, val=True, test=False)
test_label_totals= get_class_totals(all_transformed_df_test, train=False, val=False, test=True)

def plot_class_totals(train_totals, val_totals, test_totals):
    plt.figure(figsize=(14, 8), dpi=80)
    labels= ['Non-Compartmental Parameters', 'Compartmental Parameters', 'Parameter-Covariate Relationships',
             'Parameters Other', 'Doses', 'Number of Subjects',
             'Sample Timings', 'Demographics', 'Covariates Other', 'Not Relevant']
    barWidth = 0.3
    r1 = np.arange(10)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    rects1 = plt.bar(r1, train_totals['number_of_tables'], label="Train", edgecolor='white', width=barWidth)
    rects2 = plt.bar(r2, val_totals['number_of_tables'], label="Validation", edgecolor='white', width=barWidth)
    rects3 = plt.bar(r3, test_totals['number_of_tables'], label="Test", edgecolor='white', width=barWidth)
    plt.xticks([r + barWidth for r in range(len(labels))], labels, rotation=90)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.bar_label(rects1, padding=3)
    plt.bar_label(rects2, padding=3)
    plt.bar_label(rects3, padding=3)
    plt.legend()
    plt.tight_layout()
    plt.show()

#plot_class_totals(train_label_totals, val_label_totals, test_label_totals)

def plot_mutlilabel_tables(train_label_totals, val_label_totals, test_label_totals):
    train_rowsums = train_label_totals.sum(axis=1)
    x_train = train_rowsums.value_counts()
    x_train= x_train.sort_index()

    val_rowsums = val_label_totals.sum(axis=1)
    x_val = val_rowsums.value_counts()
    x_val= x_val.sort_index()

    test_rowsums = test_label_totals.sum(axis=1)
    x_test = test_rowsums.value_counts()
    x_test.loc[0] = 127
    x_test= x_test.sort_index()

    plt.figure(figsize=(14, 8), dpi=80)
    labels= [0,1,2,3,4,5]
    barWidth = 0.3
    r1 = np.arange(len(x_train.index))
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    rects1 = plt.bar(r1, x_train.values, label="Train", edgecolor='white', width=barWidth)
    rects2 = plt.bar(r2, x_val.values, label="Validation", edgecolor='white', width=barWidth)
    rects3 = plt.bar(r3, x_test.values, label="Test", edgecolor='white', width=barWidth)
    plt.bar_label(rects1, padding=3)
    plt.bar_label(rects2, padding=3)
    plt.bar_label(rects3, padding=3)
    plt.ylabel('Frequency', fontsize=12)
    plt.xlabel('Number of Labels per Table', fontsize=12)
    plt.xticks([r + barWidth for r in range(len(labels))], labels)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_mutlilabel_tables(all_transformed_df_train, all_transformed_df_val, all_transformed_df_test)


def plot_html_len(df):
    lens = df["html"].str.len()
    lens.hist(bins=np.arange(0, 5000, 50))
    plt.title("Len of html string")
    plt.ylabel('Words', fontsize=12)
    plt.xlabel('Count', fontsize=12)
    plt.tight_layout()
    plt.show()

