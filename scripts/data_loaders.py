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


def process_table_data(inp_samples: List[Dict], inp_tokenizer: str, max_len: int,
                       dataset_name: str) -> Dataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    htmls = [sample["html"] for sample in inp_samples]
    htmls = preprocess_htmls(inp_samples)

    labels = [sample["accept"] for sample in inp_samples]
    labels = convert_labels(inp_labels=labels)

    tokenizer = Tokenizer.from_file(inp_tokenizer)
    table_encodings = tokenizer(htmls, padding=True, truncation=True, max_length=max_len,
                                  return_overflowing_tokens=True)
    all_tokens = extract_all_tokens_withoutpad(table_encodings.encodings)
    print_token_stats(all_tokens=all_tokens, dataset_name=dataset_name,
                      max_len=max_len, plot_histogram=True)

    print(f"Number of sentences : {len(all_tokens)}")

    print_few_mentions(all_tokens=all_tokens, labels=labels, n=5)

    encoded_labels = pad_and_encode_labels(all_labels=labels,
                                           max_len=max_len,
                                           tag2id=tag2id)

    torch_dataset = PKDataset(encodings=doc_encodings, labels=encoded_labels)

    return torch_dataset

def extract_all_tokens_withoutpad(html_encodings: List[Encoding]):
    return [[t for t in enc.tokens if t != '[PAD]'] for enc in html_encodings]

#checked this and it works!
def convert_labels(all_labels: List[List[str]],  max_len: int):
    """Transforms labels to vector of binary numbers"""
    mlb= MultiLabelBinarizer()
    all_transformed_labels = mlb.fit_transform(all_labels)
    for i in all_transformed_labels:
        assert len(i) == max_length
    print(all_transformed_labels[1:4])
    return all_transformed_labels
