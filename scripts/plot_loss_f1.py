import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob


def get_f1_scores(f1_folder_name):
    f1_dict = {}
    file_type = 'csv'
    seperator = ','
    for f in glob.glob(f1_folder_name + "/*." + file_type):
        df = pd.read_csv(f, sep=seperator)
        df = df["Value"]
        tag = str(f).split("--tag-")
        name = tag[0].split("FFNN-")[1]
        if tag[1] == "F1_weighted_val.csv":
            f1_dict[name] = df
    sorted_f1s = {key: value for key, value in sorted(f1_dict.items())}
    f1_df = pd.DataFrame(sorted_f1s)
    cols = {'Tokenizer-10000-Nohtml': "PMC-10000-nohtml",
            'Tokenizer-10000Special': "PMC-10000",
            'Tokenizer-5000': "PMC-5000",
            'Tokenizer-5000-noHtml': "PMC-5000-nohtml",
            'Tokenizer-PK1000Special': "PK-10000",
            'Tokenizer-PK1000Special-nohtml': "PK-10000-nohtml",
            'Tokenizer-PK3000Special': "PK-3000",
            'Tokenizer-PK3000Special-Nohtml': "PK-3000-nohtml",
            'Tokenizer-PK5000Special': "PK-5000",
            'Tokenizer-PK5000Special-Nohtml': "PK-5000-nohtml"}
    f1_df = f1_df.rename(columns=cols)
    return f1_df


def get_loss_scores(loss_folder_name):
    loss_dict = {}
    file_type = 'csv'
    seperator = ','
    for f in glob.glob(loss_folder_name + "/*." + file_type):
        df = pd.read_csv(f, sep=seperator)
        df = df["Value"]
        tag = str(f).split("--tag-")
        name = tag[0].split("FFNN-")[1]
        if tag[1] == "Loss_val.csv":
            loss_dict[name] = df
    sorted_dict = {key: value for key, value in sorted(loss_dict.items())}
    loss_df = pd.DataFrame(sorted_dict)
    cols = {'Tokenizer-10000-Nohtml': "PMC-10000-nohtml",
            'Tokenizer-10000Special': "PMC-10000",
            'Tokenizer-5000': "PMC-5000",
            'Tokenizer-5000-noHtml': "PMC-5000-nohtml",
            'Tokenizer-PK1000Special': "PK-10000",
            'Tokenizer-PK1000Special-nohtml': "PK-10000-nohtml",
            'Tokenizer-PK3000Special': "PK-3000",
            'Tokenizer-PK3000Special-Nohtml': "PK-3000-nohtml",
            'Tokenizer-PK5000Special': "PK-5000",
            'Tokenizer-PK5000Special-Nohtml': "PK-5000-nohtml"}
    loss_df= loss_df.rename(columns=cols)
    return loss_df


def plot_f1(df):
    df.plot.line()
    plt.xlabel("Epochs")
    plt.ylabel("Weighted-F1 Score")
    plt.xlim(right=40)
    plt.legend()
    plt.show()


def plot_loss(df):
    df.plot.line()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(right=40)
    plt.legend()
    plt.show()


f1_folder_name = '../data/runs_csvs/tokenizers_val_f1s'
loss_folder_name = '../data/runs_csvs/tokenizers_val_loss'

f1_scores = get_f1_scores(f1_folder_name)
losses = get_loss_scores(loss_folder_name)

plot_f1(f1_scores)
plot_loss(losses)
a = 1
