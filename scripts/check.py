import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import jsonlines
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from typing import List, Dict
from transformers import PreTrainedTokenizerFast
from data_loaders.table_data_loaders import extract_all_ids_withoutpad
import re
import bs4 as bs

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print('Using device:', device)

with jsonlines.open("../data/train-test-val/train-corrected.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

htmls= [x["html"] for x in json_list[:100]]

def preprocess_htmls(inp_samples: List[str], remove_html=bool):
    """Remove style etc. from html tables"""
    processed_htmls = []
    for html in inp_samples:
        replace_n = html.replace("\n", "#n#")
        replace_style = re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
        if remove_html:
            final_html = re.sub(r"\<(?:[^<>])*\>", ' ', replace_style)
        else:
            final_html = re.sub(r'''\<table xmlns:(.*?)rules="groups"\>''', '<table>', replace_style)
        processed_htmls.append(final_html)

    return processed_htmls

def findall_colnames(table):
    column_names = []
    for row in table.find_all('tr'):
        th_tags = row.find_all('th')
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())
    return column_names

def findall_rownames(table):
    table_rows = table.find_all('tr')
    row_names = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        if row:
            row_names.append(row[0]) #
    return row_names

def findall_rows(table):
    table_rows = table.find_all('tr')
    row_names = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        if row:
            row_names.append(row[1:])
    return row_names

def join_section_info(title, column_names, row_names, all_rows):
    final_list = []
    final_list.append("[CAPTION]")
    final_list.extend(title)
    final_list.append("[FIRST_ROW]")
    #final_list.append("[SEP]")
    final_list.extend(column_names)
    final_list.append("[FIRST_COL]")
    final_list.extend(row_names)
    final_list.append("[TABLE_BODY]")
    final_list.extend(all_rows)
    final_str = ' '.join(final_list)
    return final_str

def separated_sections(inp_samples: List[str]):
    """Extracts just title, first column and first row contents from table"""
    processed_htmls = []
    for html in inp_samples:
        title = re.findall("\<h4>(.*?)\</h4>", html)
        soup = bs.BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        all_rows = findall_rows(table)
        body = [item for sublist in all_rows for item in sublist]
        row_names = findall_rownames(table)
        col_names = findall_colnames(table)
        final_html = join_section_info(title, col_names, row_names, body)
        processed_htmls.append(final_html)

    return processed_htmls

z= separated_sections(htmls)
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_tokens(["[CAPTION]", "[FIRST_ROW]", "[FIRST_COL]", "[TABLE_BODY]"])
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
table_encodings = tokenizer(z)
input_ids = table_encodings["input_ids"]
tokens = [tokenizer.convert_ids_to_tokens(x) for x in input_ids]
print(tokens[0])

a=1

