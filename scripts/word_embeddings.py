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
from tokenizers import SentencePieceBPETokenizer
from transformers import PreTrainedTokenizerFast
import matplotlib.pyplot as plt

def print_token_stats(all_tokens: List[List[str]], dataset_name: str,
                      plot_histogram: bool = False):
    n_tokens = []
    for tokens in all_tokens:
        nt = len(tokens)
        n_tokens.append(nt)
    if plot_histogram:
        plt.hist(n_tokens, bins=50)
        plt.title(f"Number of bert tokens in the {dataset_name.upper()} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()

def extract_all_ids_withoutpad(html_encodings: List[List[int]]):
    ids_list= []
    for l in html_encodings:
        no_pads= [i for i in l if i != 5000]
        ids_list.append(no_pads)
    return ids_list

def preprocess_htmls(inp_samples: List[List[str]]):
    """Remove style etc. from html tables"""
    processed_htmls= []
    for html in inp_samples:
        replace_n = html.replace("\n", "#n#")
        replace_style = re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
        final_html = re.sub(r'''\<table xmlns:xlink=(.*?)rules="groups"\>''', ' ', replace_style)
        processed_htmls.append(final_html)

    return processed_htmls

my_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]
my_list = ["<html><table><th> blah", "<html><table><th> blah <\html>", "<html><table><th> blah blah <\html>"]
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
htmls = [sample["html"] for sample in my_data]
prepro_htmls= preprocess_htmls(htmls)
encodings = tokenizer(prepro_htmls, padding=True, truncation=True, max_length=10)
ids = encodings["input_ids"]
#print(ids)

no_pads = extract_all_ids_withoutpad(encodings["input_ids"])
print(no_pads)
#print_token_stats(all_tokens=no_pads, dataset_name="test", plot_histogram=True)

