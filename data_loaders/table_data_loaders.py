from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import re
import ujson
from pathlib import Path
from typing import Dict, List, Iterable
from transformers import PreTrainedTokenizerFast
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import matplotlib
import torch
import os
import json
import bs4 as bs
from data_loaders.extract_baseline import findall_rownames, findall_colnames, join_info, join_section_info, findall_rows
import random
from torch.utils.data.sampler import Sampler

matplotlib.style.use('seaborn-whitegrid')


# my_test_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]

def convert_labels_5s(inp_labels: List[List[str]]):
    """Transforms labels to vector of binary numbers"""
    mlb = MultiLabelBinarizer()
    # remove not relevant class
    new_labels = [list(filter(lambda a: a != 4, x)) for x in inp_labels]
    all_transformed_labels = mlb.fit_transform(new_labels)
    all_transformed_labels = list(all_transformed_labels)
    # print(all_transformed_labels[1:4])
    return all_transformed_labels


def convert_labels(inp_labels: List[List[str]]):
    """Transforms labels to vector of binary numbers"""
    mlb = MultiLabelBinarizer()
    # remove not relevant class
    new_labels = inp_labels
    all_transformed_labels = mlb.fit_transform(new_labels)
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
            final_html = re.sub(r'''\<table xmlns:(.*?)rules="groups"\>''', '<table>', replace_style)

        final_html = re.sub(' +', ' ', final_html)
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


def print_token_stats(all_tokens: List[List[int]], dataset_name: str,
                      plot_histogram: bool = False):
    """Plots total tokens per table"""
    n_tokens = []
    for tokens in all_tokens:
        nt = len(tokens)
        n_tokens.append(nt)
    if plot_histogram:
        plt.hist(n_tokens, bins=50)
        #plt.title(f"Number of tokens in the {dataset_name.upper()} set")
        plt.xlabel("Number of Tokens")
        plt.ylabel("Number of Tables")
        plt.show()
        plt.close()


def extract_all_ids_withoutpad(html_encodings: List[List[int]]):
    ids_list = []
    for l in html_encodings:
        no_pads = [i for i in l if i != 5000]
        ids_list.append(no_pads)
    return ids_list


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


class PKDataset(Dataset):

    def __init__(self, encodings, labels, task_hash, input_hash):
        self.encodings = encodings
        self.labels = labels
        self.task_hash = task_hash
        self.input_hash = input_hash

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index]).type(
            torch.float32)  # torch.DoubleTensor # size [n_samples, n_labels]
        item["_task_hash"] = self.task_hash[index]
        item["_input_hash"] = self.input_hash[index]
        return item

    def __len__(self):
        return len(self.labels)


class PKDatasetMutli(Dataset):

    def __init__(self, encodings, labels, task_hash, input_hash, mutli_hot):
        self.encodings = encodings
        self.labels = labels
        self.task_hash = task_hash
        self.input_hash = input_hash
        self.multihot = mutli_hot

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[index]).type(
            torch.float32)  # torch.DoubleTensor # size [n_samples, n_labels]
        item["multi_hot"] = torch.tensor(self.multihot[index]).type(torch.float32)
        item["_task_hash"] = self.task_hash[index]
        item["_input_hash"] = self.input_hash[index]
        return item

    def __len__(self):
        return len(self.labels)


def get_multi_hot(lst, vocab):
    onehot_encoded = []
    for value in lst:
        empty = [0 for _ in range(len(vocab))]
        empty[value] = 1
        onehot_encoded.append(empty)
    summed = [sum(elts) for elts in zip(*onehot_encoded)]
    return summed


def process_table_data(inp_samples: List[Dict], inp_tokenizer: str, max_len: int,
                       dataset_name: str, remove_html: bool, baseline_only: bool, remove_5s: bool,
                       sections: bool, mutli_hot: bool, aug_all: bool, aug_syns: bool, aug_nums: bool,
                       aug_both: bool) -> PKDataset:
    """
    Generates a pytorch dataset containing encoded tokens and labels
    """

    print(f"\n==== {dataset_name.upper()} set ====")
    htmls = [sample["html"] for sample in inp_samples]

    if baseline_only:
        prepro_htmls = title_header_cols(inp_samples=htmls)
    elif sections:
        prepro_htmls = separated_sections(inp_samples=htmls)
    else:
        prepro_htmls = preprocess_htmls(inp_samples=htmls, remove_html=remove_html)

    # augmentation
    if aug_syns:
        augmented_syns = augment_syns(prepro_htmls, "../config/data_augment.json")
        prepro_htmls.extend(augmented_syns)
    elif aug_nums:
        augmented_nums = augment_nums(prepro_htmls)
        prepro_htmls.extend(augmented_nums)
    elif aug_all:
        augmented_all = augment_all(prepro_htmls)
        prepro_htmls.extend(augmented_all)
    elif aug_both:
        nums = augment_nums(prepro_htmls)
        syns = augment_syns(prepro_htmls, "../config/data_augment.json")
        prepro_htmls.extend(nums)
        prepro_htmls.extend(syns)
    else:
        prepro_htmls = prepro_htmls

    labels = [sample["accept"] for sample in inp_samples]
    if remove_5s:
        labels = convert_labels_5s(inp_labels=labels)
    else:
        labels = convert_labels(inp_labels=labels)

    task_hash = [sample["_task_hash"] for sample in inp_samples]
    input_hash = [sample["_input_hash"] for sample in inp_samples]

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=inp_tokenizer)
    # tokenizer.add_tokens(["[CAPTION]", "[FIRST_ROW]", "[FIRST_COL]", "[TABLE_BODY]"], special_tokens=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    table_encodings = tokenizer(prepro_htmls, padding=True, truncation=True, max_length=max_len)
    table_ids_no_pad = extract_all_ids_withoutpad(table_encodings["input_ids"])
    print_token_stats(all_tokens=table_ids_no_pad, dataset_name=dataset_name, plot_histogram=True)

    print(f"Number of tables : {len(table_ids_no_pad)}")

    # get mutli-hot vectors:
    if mutli_hot:
        multi_hot_vectors = [get_multi_hot(lst, vocab=tokenizer.vocab) for lst in table_ids_no_pad]
        torch_dataset = PKDatasetMutli(encodings=table_encodings, mutli_hot=multi_hot_vectors, labels=labels,
                                       task_hash=task_hash, input_hash=input_hash)
    else:
        torch_dataset = PKDataset(encodings=table_encodings, labels=labels, task_hash=task_hash, input_hash=input_hash)

    input_ids = table_encodings["input_ids"]
    check_tokens = [tokenizer.convert_ids_to_tokens(x) for x in input_ids]
    print(check_tokens[1])
    print(table_encodings.input_ids[1])
    print(labels[1])

    return torch_dataset


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: str, max_len: int,
                    shuffle: bool, sampler: bool, n_workers: int, dataset_name: str, remove_html: bool,
                    baseline_only: bool, remove_5s: bool, sections: bool, multi_hot: bool,
                    aug_all: bool, aug_syns: bool, aug_nums: bool, aug_both: bool
                    ) -> [DataLoader, PKDataset]:
    torch_dataset = process_table_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                       dataset_name=dataset_name, remove_html=remove_html, baseline_only=baseline_only,
                                       remove_5s=remove_5s, sections=sections, mutli_hot=multi_hot,
                                       aug_all=aug_all, aug_syns=aug_syns, aug_nums=aug_nums, aug_both=aug_both)

    if sampler:
        train_idx = list(range(len(torch_dataset)))
        under_sampler = MultilabelBalancedRandomSampler(torch_dataset, train_idx, class_choice="least_sampled")
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, sampler=under_sampler)

    elif shuffle:
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


def repl1(match):
    matched = match.group(0)
    if '.' or 'e' in matched.lower():
        num = round(float(matched) * 0.9, 1)
    else:
        num = int(matched) + 1
    return str(num)


def repl2(match):
    matched = match.group(0)
    if '.' or 'e' in matched.lower():
        num = round(float(matched) * 1.1, 1)
    else:
        num = int(matched) - 1
    return str(num)


def augment_nums(inp_samples: List[str]):
    """Agitates numbers by 10% up or down"""
    augmented_samples = []
    for sample in inp_samples:
        if random.choice([0, 1]) < 0.5:
            aug_html = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl1, sample)
        else:
            aug_html = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl2, sample)

        augmented_samples.append(aug_html)
    return augmented_samples


def get_match_pattern(af):
    all_matches = {}
    for dic in af:
        match_patterns = {}
        for subdic in af[dic]:
            match_lst = af[dic][subdic]
            match_lst.insert(0, "abuffer")
            match_lst.append("zbuffer")
            match_lst = [x.lower() for x in match_lst]
            match_pattern = " | ".join(sorted(re.escape(x) for x in match_lst))
            match_patterns[subdic] = match_pattern
            all_matches.update(match_patterns)
    return all_matches


def grab_children(af):
    dict_list = []
    local_list = []
    for key, value in af.items():
        dict_list.append(value)
    for item in dict_list:
        for k, v in item.items():
            local_list.append(v)
    return local_list


def repl_syns(match):
    matched = str(match.group(0)).lower().strip()

    with open("../config/data_augment.json") as f:
        af = json.load(f)

    # list of replacement lists
    match_lsts = grab_children(af)
    match_lsts = [[x.lower().strip() for x in lst] for lst in match_lsts]
    match_lst = [lst for lst in match_lsts if matched in lst]
    match_lst = [item for sublist in match_lst for item in sublist]
    if matched in match_lst:
        replace_lst = [n for n in match_lst if n != matched]
        value = random.choice(replace_lst)
    else:
        print("error")
        value = matched
    value = value.center((len(value) + 2))
    return str(value)


def augment_syns(inp_samples: List[str], aug_file):
    """Replaces PK terms and units with synonyms """

    with open(aug_file) as f:
        af = json.load(f)
    match_patterns = get_match_pattern(af)

    augmented_htmls = inp_samples
    for k, v in match_patterns.items():
        augmented_htmls = [re.sub(match_patterns[k], repl_syns, html, flags=re.IGNORECASE) for html in augmented_htmls]

    return augmented_htmls


def augment_all(inp_samples: List[str]):
    """Agitates numbers by 10% up or down or replaces PK terms with synonyms """
    syn_samples = augment_syns(inp_samples, "../config/data_augment.json")
    combined_samples = augment_nums(syn_samples)
    return combined_samples


def get_dataloaders(inp_data_dir: str, inp_tokenizer: str, max_len: int,
                    batch_size: int, val_batch_size: int, n_workers: int, remove_html: bool, baseline_only: bool,
                    aug_all: bool, aug_nums: bool, aug_syns: bool, aug_both: bool, sampler: bool, sections: bool,
                    multi_hot: bool) -> \
        Iterable[DataLoader]:
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
    train_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="train-corrected"))
    valid_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="val-corrected"))
    test_samples = list(read_dataset(data_dir_inp=inp_data_dir, dataset_name="test-corrected"))

    '''
    if aug_all:
        augmented_samples = augment_all(train_samples)
        train_samples.extend(augmented_samples)
    elif aug_nums:
        augmented_samples = augment_nums(train_samples)
        train_samples.extend(augmented_samples)
    elif aug_syns:
        augmented_samples = augment_syns(train_samples, "../config/data_augment.json")
        train_samples.extend(augmented_samples)
    elif aug_both:
        syns_samples = augment_syns(train_samples, "../config/data_augment.json")
        nums_samples = augment_nums(train_samples)
        train_samples.extend(syns_samples)
        train_samples.extend(nums_samples)
    else:
        train_samples = train_samples
    '''

    print(f"Total Length of Train Samples is: {len(train_samples)}")

    train_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                      "_input_hash": dic["_input_hash"]} for dic in train_samples]
    valid_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                      "_input_hash": dic["_input_hash"]} for dic in valid_samples]
    test_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                     "_input_hash": dic["_input_hash"]} for dic in test_samples]

    train_dataloader, train_dataset = make_dataloader(inp_samples=train_samples, batch_size=batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=True, sampler=sampler,
                                                      n_workers=n_workers,
                                                      dataset_name='training', remove_html=remove_html,
                                                      baseline_only=baseline_only,
                                                      remove_5s=False, sections=sections, multi_hot=multi_hot,
                                                      aug_all=aug_all, aug_syns=aug_syns, aug_nums=aug_nums,
                                                      aug_both=aug_both)
    valid_dataloader, valid_dataset = make_dataloader(inp_samples=valid_samples, batch_size=val_batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=False, sampler=False,
                                                      n_workers=n_workers,
                                                      dataset_name='dev', remove_html=remove_html,
                                                      baseline_only=baseline_only,
                                                      remove_5s=False, sections=sections, multi_hot=multi_hot,
                                                      aug_all=False, aug_syns=False, aug_nums=False, aug_both=False)
    test_dataloader, test_dataset = make_dataloader(inp_samples=test_samples, batch_size=val_batch_size,
                                                    inp_tokenizer=inp_tokenizer,
                                                    max_len=max_len, shuffle=False, sampler=False, n_workers=n_workers,
                                                    dataset_name='test', remove_html=remove_html,
                                                    baseline_only=baseline_only,
                                                    remove_5s=True, sections=sections, multi_hot=multi_hot,
                                                    aug_all=False, aug_syns=False, aug_nums=False, aug_both=False)

    return train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset


class MultilabelBalancedRandomSampler(Sampler):
    """
    MultilabelBalancedRandomSampler: Given a multilabel dataset of length n_samples and
    number of classes n_classes, samples from the data with equal probability per class
    effectively oversampling minority classes and undersampling majority classes at the
    same time. Note that using this sampler does not guarantee that the distribution of
    classes in the output samples will be uniform, since the dataset is multilabel and
    sampling is based on a single class. This does however guarantee that all classes
    will have at least batch_size / n_classes samples as batch_size approaches infinity
    """

    def __init__(self, dataset, indices=None, class_choice="least_sampled"):
        """
        Parameters:
        -----------
            labels: a multi-hot encoding numpy array of shape (n_samples, n_classes)
            indices: an arbitrary-length 1-dimensional numpy array representing a list
            of indices to sample only from
            class_choice: a string indicating how class will be selected for every
            sample:
                "least_sampled": class with the least number of sampled labels so far
                "random": class is chosen uniformly at random
                "cycle": the sampler cycles through the classes sequentially
        """
        self.labels = torch.stack([sample["labels"] for sample in dataset])
        self.indices = indices
        if self.indices is None:
            self.indices = range(len(self.labels))

        self.num_classes = self.labels.shape[1]

        # List of lists of example indices per class
        self.class_indices = []
        for class_ in range(self.num_classes):
            lst = np.where(self.labels[:, class_] == 1)[0]
            lst = lst[np.isin(lst, self.indices)]
            self.class_indices.append(lst)

        self.counts = [0] * self.num_classes

        assert class_choice in ["least_sampled", "random", "cycle"]
        self.class_choice = class_choice
        self.current_class = 0

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count >= len(self.indices):
            raise StopIteration
        self.count += 1
        return self.sample()

    def sample(self):
        class_ = self.get_class()
        class_indices = self.class_indices[class_]
        chosen_index = np.random.choice(class_indices)
        if self.class_choice == "least_sampled":
            for class_, indicator in enumerate(self.labels[chosen_index]):
                if indicator == 1:
                    self.counts[class_] += 1
        return chosen_index

    def get_class(self):
        if self.class_choice == "random":
            class_ = random.randint(0, self.labels.shape[1] - 1)
        elif self.class_choice == "cycle":
            class_ = self.current_class
            self.current_class = (self.current_class + 1) % self.labels.shape[1]
        elif self.class_choice == "least_sampled":
            min_count = self.counts[0]
            min_classes = [0]
            for class_ in range(1, self.num_classes):
                if self.counts[class_] < min_count:
                    min_count = self.counts[class_]
                    min_classes = [class_]
                if self.counts[class_] == min_count:
                    min_classes.append(class_)
            class_ = np.random.choice(min_classes)
        return class_

    def __len__(self):
        return len(self.indices)
