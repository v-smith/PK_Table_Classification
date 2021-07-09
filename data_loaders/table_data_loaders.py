from sklearn.preprocessing import MultiLabelBinarizer
import re
import ujson
from pathlib import Path
from typing import Dict, List, Iterable
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import bs4 as bs
from data_loaders.extract_baseline import findall_rownames, findall_colnames, join_info

matplotlib.style.use('ggplot')
# my_test_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]

def convert_labels(inp_labels: List[List[str]]):
    """Transforms labels to vector of binary numbers"""
    mlb = MultiLabelBinarizer()
    all_transformed_labels = mlb.fit_transform(inp_labels)
    all_transformed_labels = list(all_transformed_labels)
    # print(all_transformed_labels[1:4])
    return all_transformed_labels


def preprocess_htmls(inp_samples: List[str], remove_html=bool):
    """Remove style etc. from html tables"""
    processed_htmls = []
    for html in inp_samples:
        replace_n = html.replace("\n", "#n#")
        replace_style = re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
        if remove_html:
            final_html = re.sub(r"\<(?:[^<>])*\>", ' ', replace_style)
        else:
            final_html = re.sub(r'''\<table xmlns:(.*?)rules="groups"\>''', ' ', replace_style)
        processed_htmls.append(final_html)

    return processed_htmls

def title_header_cols(inp_samples: List[str]):
    """Extracts just title, first column and first row contents from table"""
    processed_htmls = []
    for html in inp_samples:
        title = re.findall("\<h4>(.*?)\</h4>", html)
        soup = bs.BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        row_names = findall_rownames(table)
        col_names = findall_colnames(table)
        final_html = join_info(title, col_names, row_names)
        processed_htmls.append(final_html)

    return processed_htmls


def augment_tables(inp_samples: List[str]):
    """Replaces PK terms with synonyms and obliterates some rows etc."""
    # dictionary of terms
    return inp_samples


def print_token_stats(all_tokens: List[List[int]], dataset_name: str,
                      plot_histogram: bool = False):
    """Plots total tokens per table"""
    n_tokens = []
    for tokens in all_tokens:
        nt = len(tokens)
        n_tokens.append(nt)
    if plot_histogram:
        plt.hist(n_tokens, bins=50)
        plt.title(f"Number of tokens in the {dataset_name.upper()} set")
        plt.xlabel("# tokens")
        plt.ylabel("# sentences")
        plt.show()
        plt.close()


def extract_all_ids_withoutpad(html_encodings: List[List[int]]):
    ids_list = []
    for l in html_encodings:
        no_pads = [i for i in l if i != 5000]
        ids_list.append(no_pads)
    return ids_list


class PKDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index]).type(torch.float32)  #torch.DoubleTensor # size [n_samples, n_labels]
        return item

    def __len__(self):
        return len(self.labels)


def process_table_data(inp_samples: List[Dict], inp_tokenizer: str, max_len: int,
                       dataset_name: str, remove_html: bool, baseline_only: bool) -> PKDataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    htmls = [sample["html"] for sample in inp_samples]
    if baseline_only:
        prepro_htmls = title_header_cols(inp_samples=htmls)
    else:
        prepro_htmls = preprocess_htmls(inp_samples=htmls, remove_html=remove_html)

    labels = [sample["accept"] for sample in inp_samples]
    labels = convert_labels(inp_labels=labels)

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=inp_tokenizer)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    table_encodings = tokenizer(prepro_htmls, padding=True, truncation=True, max_length=max_len)
    table_ids_no_pad = extract_all_ids_withoutpad(table_encodings["input_ids"])
    print_token_stats(all_tokens=table_ids_no_pad, dataset_name=dataset_name, plot_histogram=True)

    print(f"Number of tables : {len(table_ids_no_pad)}")

    check_tokenizer = Tokenizer.from_file(inp_tokenizer)
    check_tokens = check_tokenizer.encode(prepro_htmls[1]).tokens
    print(check_tokens)

    print(table_encodings.input_ids[1])
    print(labels[1])

    torch_dataset = PKDataset(encodings=table_encodings, labels=labels)
    return torch_dataset


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: str, max_len: int,
                    shuffle: bool, n_workers: int, dataset_name: str, remove_html:bool, baseline_only:bool) -> [DataLoader, PKDataset]:
    torch_dataset = process_table_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                       dataset_name=dataset_name, remove_html=remove_html, baseline_only=baseline_only)
    if shuffle:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    else:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers)

    return loader, torch_dataset


def read_jsonl(file_path):
    # Taken from prodigy support
    """Read a .jsonl file and yield its contents line by line.
    file_path (unicode / Path): The file path.
    YIELDS: The loaded JSON contents of each line.
    """
    with Path(file_path).open(encoding='utf8') as f:
        for line in f:
            try:  # hack to handle broken jsonl
                yield ujson.loads(line.strip())
            except ValueError:
                continue


def read_dataset(data_dir_inp: str, dataset_name: str):
    file_name = f"{dataset_name}.jsonl"
    file_path = os.path.join(data_dir_inp, file_name)
    json_list = read_jsonl(file_path=file_path)
    return json_list


def get_dataloaders(inp_data_dir: str, inp_tokenizer: str, max_len: int,
                    batch_size: int, val_batch_size: int, n_workers: int, remove_html:bool, baseline_only:bool) -> Iterable[DataLoader]:
    """
    Function to generate data loaders
    @param inp_data_dir: Input data directory expecting 3 files: train.jsonl and valid.jsonl and test.jsonl
    @param inp_tokenizer: Pre-trained Tokenizer from file
    @param max_len: maximum length of token list
    @param batch_size: batch size
    @param val_batch_size: batch size for the validation/test datasets
    @param n_workers: number of workers for the dataloader
    @return: pytorch data loaders
    """
    train_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="train"))
    valid_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="val"))
    test_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="test"))

    train_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in train_samples]
    valid_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in valid_samples]
    test_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in test_samples]

    train_dataloader, train_dataset = make_dataloader(inp_samples=train_samples, batch_size=batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=True, n_workers=n_workers,
                                                      dataset_name='training', remove_html=remove_html, baseline_only=baseline_only)
    valid_dataloader, valid_dataset = make_dataloader(inp_samples=valid_samples, batch_size=val_batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=False, n_workers=n_workers,
                                                      dataset_name='dev', remove_html=remove_html, baseline_only=baseline_only)
    test_dataloader, test_dataset = make_dataloader(inp_samples=test_samples, batch_size=val_batch_size,
                                                    inp_tokenizer=inp_tokenizer,
                                                    max_len=max_len, shuffle=False, n_workers=n_workers,
                                                    dataset_name='test', remove_html=remove_html, baseline_only=baseline_only)

    return train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset



