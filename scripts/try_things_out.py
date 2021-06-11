import jsonlines
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import re
import ujson
from pathlib import Path
from typing import Dict, List, Iterable
import numpy as np
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tokenizers import Encoding

my_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]

def convert_labels(inp_labels: List[List[str]], max_len: int):
    """Transforms labels to vector of binary numbers"""
    mlb= MultiLabelBinarizer()
    all_transformed_labels = mlb.fit_transform(inp_labels)
    for i in all_transformed_labels:
        assert len(i) == max_len
    print(all_transformed_labels[1:4])
    return all_transformed_labels

def preprocess_htmls(inp_samples: List[List[str]]):
    """Remove style etc. from html tables"""
    processed_htmls= []
    for html in inp_samples:
        replace_n = html.replace("\n", "#n#")
        replace_style = re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
        processed_html = re.sub(r'''\<table xmlns:xlink=(.*?)rules="groups"\>''', ' ', replace_style)
        processed_htmls.append(processed_html)

    return processed_html


def process_table_data(inp_samples: List[Dict], inp_tokenizer: str, max_len: int,
                       dataset_name: str) -> Dataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    htmls = [sample["html"] for sample in inp_samples]
    htmls = preprocess_htmls(inp_samples=htmls)

    labels = [sample["accept"] for sample in inp_samples]
    labels = convert_labels(inp_labels=labels, max_len=5)

    tokenizer = Tokenizer.from_file(inp_tokenizer)
    for table in htmls:
        table_encoding = tokenizer(table, padding=True, truncation=True, max_length=max_len)

    all_tokens =table_encodings.ids
    #print_token_stats(all_tokens=all_tokens, dataset_name=dataset_name,
                      #max_len=max_len, plot_histogram=True)

    print(f"Number of tables : {len(all_tokens)}")

    print(all_tokens[1:4])
    print(labels[1:4])

    torch_dataset = PKDataset(encodings=table_encodings, labels=labels)

    return torch_dataset

def extract_all_tokens_withoutpad(html_encodings: List[Encoding]):
    return [[t for t in enc.tokens if t != '[PAD]'] for enc in html_encodings]




process_table_data(my_data, "../tokenizers/tokenizerPKtablesSpecialTokens5000.json", 600, "test")


