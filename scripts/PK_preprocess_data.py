import jsonlines
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re
import ujson
from pathlib import Path

def read_jsonl(file_path):
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open('r', encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue

def write_to_text(input_json, out_file):
    all_html = [x['html'].replace("\n", "#n#") for x in read_jsonl(file_path=input_json)]
    with open(out_file, 'w') as f:
        for i in all_html:
            i= re.sub(r"\<style>(.*?)\</style>", ' ', i)
            i =re.sub(r'''\<table xmlns:xlink=(.*?)rules="groups"\>''', ' ', i)
            f.write("%s\n" % i)

#preprocessing, remove style and sub line breaks for #n# to fit with tokenizer training
def replace_dict_values(k, l):
    new_l = []
    for dic in l:
        for key, value in dic.items():
            if key == k:
                replace_n= value.replace("\n", "#n#")
                replace_style= re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
                final_html= re.sub(r'''\<table xmlns:xlink=(.*?)rules="groups"\>''', ' ', replace_style)
                dic.update({key: final_html})
                new_l.append(dic)
    return new_l





def preprocess_data(input_file, output_labels, output_html):
    #import jsonfile into a list
    with jsonlines.open(input_file) as reader:
        json_list = []
        for obj in reader:
            json_list.append(obj)
    unique_list = list({v['text']: v for v in json_list}.values())

    preprocessed_list= replace_dict_values("html", unique_list)

    #create a df
    df = pd.DataFrame(preprocessed_list, columns=['html', 'accept'])
    df1= df.set_index("html", drop=True)

    #covert list of labels to one hot encoded
    mlb= MultiLabelBinarizer()
    label_df= pd.DataFrame(mlb.fit_transform(df1['accept']), index=df1.index, columns= mlb.classes_)
    label_df= label_df.reset_index(drop=False)
    print(label_df.head())

    #save
    label_df.to_csv(output_labels)
    write_to_text(input_file, output_html)

#run
preprocess_data("../data/final-test-covs200A.jsonl", "../data/final-test-covs200A.csv", "../data/final-test-covs200Ahtml.txt")
