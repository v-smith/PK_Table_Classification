import glob

import matplotlib.pyplot as plt
import pandas as pd

tokenizer_cols = {'Tokenizer-10000-Nohtml': "PMC-10000-nohtml",
                  'Tokenizer-10000Special': "PMC-10000",
                  'Tokenizer-5000': "PMC-5000",
                  'Tokenizer-5000-noHtml': "PMC-5000-nohtml",
                  'Tokenizer-PK1000Special': "PK-10000",
                  'Tokenizer-PK1000Special-nohtml': "PK-10000-nohtml",
                  'Tokenizer-PK3000Special': "PK-3000",
                  'Tokenizer-PK3000Special-Nohtml': "PK-3000-nohtml",
                  'Tokenizer-PK5000Special': "PK-5000",
                  'Tokenizer-PK5000Special-Nohtml': "PK-5000-nohtml"}

regions_cols = {'1000tokens-html-': "1000 tokens", '1000tokens-nohtml-': "1000 tokens, no html",
                '100tokens-nohtml': "100 tokens, no html", '2000tokens-html-': "2000 tokens",
                '200tokens-Nohtml-': "200 tokens, no html", '300tokens-nohtml-': "300 tokens, no html",
                '500tokens-html-': "500 tokens", 'Headerrowcol-100tokens-': "100 tokens, baseline",
                'Tokenizer-PK5000Special-Nohtml-': "500 tokens, no html",
                'headerrowcol-300tokens-': "300 tokens, baseline", 'sections-500tokens-': "500 tokens, Sections"}

regions_only = {'500tokens-html-': "500 tokens", 'sections-500tokens-': "500 tokens, Sections",
                'Tokenizer-PK5000Special-Nohtml-': "500 tokens,no html",
                'Headerrowcol-100tokens-': "100 tokens, baseline"}

reps_cols = {'Tokenizer-PK5000Special-Nohtml': 'Embeddings', "BOW": "BOW"}


def get_f1_scores(f1_folder_name, cols):
    f1_dict = {}
    file_type = 'csv'
    seperator = ','
    for f in glob.glob(f1_folder_name + "/*." + file_type):
        df = pd.read_csv(f, sep=seperator)
        df = df["Value"]
        tag = str(f).split("-tag-")
        name = tag[0].split("FFNN-")[1]
        if tag[1] == "F1_weighted_train.csv":
            f1_dict[name] = df
    sorted_f1s = {key: value for key, value in sorted(f1_dict.items())}
    f1_df = pd.DataFrame(sorted_f1s)
    f1_df = f1_df.rename(columns=cols)
    f1_df = f1_df[["500 tokens", "500 tokens, no html", "500 tokens, Sections", "100 tokens, baseline"]]
    f1_df.columns = ["Style Removed", "HTML Removed", "HTML Removed & Regions Marked", "Baseline"]
    return f1_df


def get_loss_scores(loss_folder_name, cols):
    loss_dict = {}
    file_type = 'csv'
    seperator = ','
    for f in glob.glob(loss_folder_name + "/*." + file_type):
        df = pd.read_csv(f, sep=seperator)
        df = df["Value"]
        tag = str(f).split("-tag-")
        name = tag[0].split("FFNN-")[1]
        if tag[1] == "Loss_train.csv":
            loss_dict[name] = df
    sorted_dict = {key: value for key, value in sorted(loss_dict.items())}
    loss_df = pd.DataFrame(sorted_dict)
    loss_df = loss_df.rename(columns=cols)
    loss_df = loss_df[["500 tokens", "500 tokens, no html", "500 tokens, Sections", "100 tokens, baseline"]]
    loss_df.columns = ["Style Removed", "HTML Removed", "HTML Removed & Regions Marked", "Baseline"]
    return loss_df


def plot_f1(df):
    df.plot.line()
    plt.xlabel("Epochs")
    plt.ylabel("Weighted-F1 Score")
    plt.xlim(right=40)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_loss(df):
    df.plot.line()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xlim(right=40)
    plt.legend()
    plt.tight_layout()
    plt.show()


f1_folder_name = '../data/runs_csvs/regions_train_f1'
# f1_folder_name = "../data/runs_csvs/BOW_val_f1"
loss_folder_name = '../data/runs_csvs/regions_train_loss'

f1_scores = get_f1_scores(f1_folder_name, regions_cols)
losses = get_loss_scores(loss_folder_name, regions_cols)

plot_f1(f1_scores)
plot_loss(losses)
