from typing import Dict, List, Iterable
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from tokenizers import Encoding
import torch
import matplotlib.pyplot as plt
import random

def get_dataloaders(inp_data_dir: str, tokenizer: str, max_len: int,
                    batch_size: int, val_batch_size: int, n_workers: int) -> Iterable[DataLoader]:
    """
    Function to generate data loaders
    @param inp_data_dir: Input data directory expecting 2 files: train.jsonl and valid.jsonl
    @param tokenizer: Pre-trained Tokenizer from file
    @param max_len: maximum length of token list
    @param batch_size: batch size
    @param val_batch_size: batch size for the validation/test datasets
    @param n_workers: number of workers for the dataloader
    @return: pytorch data loaders
    """
    train_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="training"))
    valid_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="val"))
    test_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="test"))

    all_samples = train_samples + valid_samples + test_samples
    #make labels into a binary vector here??
    unique_tags = set(tag for dic in all_samples for tag in dic["accept"])
    tag2id = {tag: tag_id for tag_id, tag in enumerate(unique_tags)}
    tag2id["PAD"] = -100  # add padding label to -100 so it can be converted

    train_samples = [{"html": dic["html"], "labels":dic["accept"]} for dic in train_samples]
    valid_samples = [{"html": dic["html"], "labels":dic["accept"]} for dic in valid_samples]
    test_samples = [{"html": dic["html"], "labels":dic["accept"]} for dic in test_samples]

    train_dataloader = make_dataloader(inp_samples=train_samples, batch_size=batch_size, inp_tokenizer=tokenizer,
                                       max_len=max_len, shuffle=True, n_workers=n_workers,
                                      dataset_name='training')
    valid_dataloader = make_dataloader(inp_samples=valid_samples, batch_size=val_batch_size, inp_tokenizer=tokenizer,
                                       max_len=max_len, shuffle=False, n_workers=n_workers,
                                       dataset_name='dev')
    test_dataloader = make_dataloader(inp_samples=test_samples, batch_size=val_batch_size, inp_tokenizer=tokenizer,
                                      max_len=max_len, shuffle=False, n_workers=n_workers,
                                      dataset_name='test')

    return train_dataloader, valid_dataloader, test_dataloader


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: str, max_len: int,
                    shuffle: bool, n_workers: int, labels: list,
                    dataset_name: str) -> DataLoader:
    torch_dataset = process_html_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                         labels=labels, dataset_name=dataset_name)
    if shuffle:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    else:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers)

    return loader


#checked this and works
def extract_all_ids_withoutpad(html_encodings: List[Encoding]):
    ids_list= []
    for l in html_encodings:
        no_pads= [i for i in l if i != 5000]
        ids_list.append(no_pads)
    return ids_list

#checked this and it works!
def convert_labels(all_labels: List[List[str]],  max_len: int):
    """Transforms labels to vector of binary numbers"""
    mlb= MultiLabelBinarizer()
    all_transformed_labels = mlb.fit_transform(all_labels)
    for i in all_transformed_labels:
        assert len(i) == max_length
    print(all_transformed_labels[1:4])
    return all_transformed_labels

#checked this and works
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
