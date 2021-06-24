from sklearn.preprocessing import MultiLabelBinarizer
import re
import ujson
from pathlib import Path
from typing import Dict, List, Iterable
from torch import nn
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer
import matplotlib.pyplot as plt
import matplotlib
import torch
import os

matplotlib.style.use('ggplot')
# my_test_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]

def convert_labels(inp_labels: List[List[str]]):
    """Transforms labels to vector of binary numbers"""
    mlb = MultiLabelBinarizer()
    all_transformed_labels = mlb.fit_transform(inp_labels)
    all_transformed_labels = list(all_transformed_labels)
    # print(all_transformed_labels[1:4])
    return all_transformed_labels


def preprocess_htmls(inp_samples: List[str]):
    """Remove style etc. from html tables"""
    processed_htmls = []
    for html in inp_samples:
        replace_n = html.replace("\n", "#n#")
        replace_style = re.sub(r"\<style>(.*?)\</style>", ' ', replace_n)
        # remove html
        # final_html = re.sub(r"\<(?:[^<>])*\>", ' ', replace_style)
        # remove link info
        final_html = re.sub(r'''\<table xmlns:xlink=(.*?)rules="groups"\>''', ' ', replace_style)
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
                       dataset_name: str) -> PKDataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    htmls = [sample["html"] for sample in inp_samples]
    prepro_htmls = preprocess_htmls(inp_samples=htmls)

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
                    shuffle: bool, n_workers: int, dataset_name: str) -> [DataLoader, PKDataset]:
    torch_dataset = process_table_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                       dataset_name=dataset_name)
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
                    batch_size: int, val_batch_size: int, n_workers: int) -> Iterable[DataLoader]:
    """
    Function to generate data loaders
    @param inp_data_dir: Input data directory expecting 2 files: train.jsonl and valid.jsonl
    @param inp_tokenizer: Pre-trained Tokenizer from file
    @param max_len: maximum length of token list
    @param batch_size: batch size
    @param val_batch_size: batch size for the validation/test datasets
    @param n_workers: number of workers for the dataloader
    @return: pytorch data loaders
    """
    train_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="builder_train800"))
    valid_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="builder_val100"))
    test_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="builder_test100"))

    train_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in train_samples]
    valid_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in valid_samples]
    test_samples = [{"html": dic["html"], "accept": dic["accept"]} for dic in test_samples]

    train_dataloader, train_dataset = make_dataloader(inp_samples=train_samples, batch_size=batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=True, n_workers=n_workers,
                                                      dataset_name='training')
    valid_dataloader, valid_dataset = make_dataloader(inp_samples=valid_samples, batch_size=val_batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=False, n_workers=n_workers,
                                                      dataset_name='dev')
    test_dataloader, test_dataset = make_dataloader(inp_samples=test_samples, batch_size=val_batch_size,
                                                    inp_tokenizer=inp_tokenizer,
                                                    max_len=max_len, shuffle=False, n_workers=n_workers,
                                                    dataset_name='test')

    return train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset


# Fully connected neural network with one hidden layer
class NeuralNet(nn.Module):
    def __init__(self, num_classes, hidden_size, embeds_size, vocab_size, padding_idx):
        super(NeuralNet, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embeds_size, padding_idx=padding_idx)
        self.l1 = nn.Linear(embeds_size, hidden_size)  # or number of classes if only one layer
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        #self.double() #if feeding in a Torch.Double type

    def forward(self, x):
        embeddings = self.embedding(x)
        max_pooled = torch.max(embeddings, dim=1, keepdim=False)[0]
        out_l1 = self.l1(max_pooled)
        out_relu = self.relu(out_l1)
        out_l2 = self.l2(out_relu)
        # no activation and no softmax at the end
        return out_l2
