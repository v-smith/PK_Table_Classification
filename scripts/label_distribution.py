import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import jsonlines
from sklearn.preprocessing import MultiLabelBinarizer


matplotlib.style.use('ggplot')


with jsonlines.open("../data/train-test-val/builder_train800.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

df= pd.DataFrame(json_list)
labels= list(df["accept"])

mlb = MultiLabelBinarizer()
all_transformed_labels = mlb.fit_transform(labels)

all_transformed_df= pd.DataFrame(all_transformed_labels)

def get_class_totals(dummy_all):
    labels = list(dummy_all.columns.values)
    counts = []
    for i in labels:
        counts.append((i, dummy_all[i].sum()))
    df_totals = pd.DataFrame(counts, columns=['label', 'number_of_tables'])
    return df_totals

label_totals= get_class_totals(all_transformed_df)

def plot_class_totals(df_totals):
    df_totals.plot(x='label', y='number_of_tables', kind='bar', legend=False, figsize=(8, 5))
    plt.title("Number of tables per class")
    plt.ylabel('# of Occurrences', fontsize=12)
    plt.xlabel('Class', fontsize=12)
    plt.xticks([0,1,2,3,4,5,6,7,8,9], ['NC Params', 'Comp Params', 'Param-Cov Relations',
                      'Params Other', 'Doses', 'Number of Subs',
                      'Sample timings', 'Demographics', 'Covariates Other', 'Not Relevant'])
    plt.tight_layout()
    plt.show()

plot_class_totals(label_totals)


#print(sns.countplot(data=all_transformed_df))
