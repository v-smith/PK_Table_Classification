from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
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
from random import choices
import random
from torch.utils.data.sampler import Sampler
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

matplotlib.style.use('ggplot')


# my_test_data= [{"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3,4,5]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1,2,3]}, {"html":"<html><table><th> blah blah <\html>", "accept":[1]}]


def convert_labels_5s(inp_labels: List[List[str]]):
    """Transforms labels to vector of binary numbers"""
    mlb = MultiLabelBinarizer()
    # remove not relevant class
    new_labels = [list(filter(lambda a: a != 5, x)) for x in inp_labels]
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


def process_table_data(inp_samples: List[Dict], inp_tokenizer: str, max_len: int,
                       dataset_name: str, remove_html: bool, baseline_only: bool, remove_5s: bool) -> PKDataset:
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
    if remove_5s:
        labels = convert_labels_5s(inp_labels=labels)
    else:
        labels = convert_labels(inp_labels=labels)

    task_hash = [sample["_task_hash"] for sample in inp_samples]
    input_hash = [sample["_input_hash"] for sample in inp_samples]

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

    torch_dataset = PKDataset(encodings=table_encodings, labels=labels, task_hash=task_hash, input_hash=input_hash)
    return torch_dataset


def make_dataloader(inp_samples: List[Dict], batch_size: int, inp_tokenizer: str, max_len: int,
                    shuffle: bool, sampler: bool, n_workers: int, dataset_name: str, remove_html: bool, baseline_only: bool,
                    remove_5s: bool) -> [DataLoader, PKDataset]:
    torch_dataset = process_table_data(inp_samples=inp_samples, inp_tokenizer=inp_tokenizer, max_len=max_len,
                                       dataset_name=dataset_name, remove_html=remove_html, baseline_only=baseline_only,
                                       remove_5s=remove_5s)

    train_idx = list(range(len(torch_dataset)))
    under_sampler = MultilabelBalancedRandomSampler(torch_dataset, train_idx, class_choice="least_sampled")

    if sampler:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, sampler=under_sampler)

    elif shuffle:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers, shuffle=True)

    else:
        loader = DataLoader(torch_dataset, batch_size=batch_size, num_workers=n_workers)

    return loader, torch_dataset


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
            self.indices = range(len(labels))

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
    # if random.random() <= 0.5:
    num = round(float(matched) * 0.9, 1)
    return str(num)


def repl2(match):
    matched = match.group(0)
    num = round(float(matched) * 1.1, 1)
    return str(num)


def repl3(match):
    matched = match.group(0)
    num = round(float(matched) * 1.2, 1)
    return str(num)


def repl4(match):
    matched = match.group(0)
    num = round(float(matched) * 0.8, 1)
    return str(num)


def augment_nums(inp_samples: List[Dict]):
    """Agitates numbers by 10% up or down, Please note input and task hashes will need to be reset at the end if this function is used """
    augmented_samples = []
    for sample in inp_samples:
        labels = sample["accept"]
        task_hash = sample["_task_hash"]
        input_hash = sample["_input_hash"]
        html_down = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl1, sample["html"])
        html_down_20 = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl4, sample["html"])
        html_up_20 = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl3, sample["html"])
        html_up = re.sub(r'[-+]?([0-9]*\.?[0-9]+)(?:[eE][-+]?[0-9]+)?', repl2, sample["html"])
        augmented_samples.append(
            {"html": html_down, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})
        augmented_samples.append(
            {"html": html_up, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})
        augmented_samples.append(
            {"html": html_down_20, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})
        augmented_samples.append(
            {"html": html_up_20, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})

    return augmented_samples


def augment_syns(inp_samples: List[Dict]):
    """Replaces PK terms and units with synonyms, Please note input and task hashes will need to be reset at the end if this function is used """
    augmented_samples = []
    # https://onlinelibrary.wiley.com/doi/full/10.1046/j.1365-2885.2001.00340.x
    units_dict = {"μg/mL": "ng/mL", "mg": "g", "min": "hr", "min-1": "hr-1", "mL/kg": "L/kg", "mL/min": "mL/h",
                  "μg · h/mL": "μg × h/mL", "%": "percent", "mg/kg": "μg/kg", "days": "hrs", "day": "week"}
    terms_dict = {"Cmax": "C p,ss(max)", "Cavg": "C p(avg)", "C min": "C p(last)", "C p(12h)": "C p(0)",
                  "t 1/2": "t 1/2(a)", "tmax": "t 1/2(d)", "MRT": "MAT",
                  "k 12": "k 21", "k el": "k a", "V D": "V c", "V D(area)": "V (d(ss))",
                  "AUC": "AUC0–12", "Area under the curve": "AUC0-24", "F": "f el", "Sample Timings": "Times",
                  "n=": "number of subjects", "fed": "fasted", "IV": "SC", "SC": "IM", "IM": "IV",
                  "Renal Impairment": "RI",
                  "Emax": "EC50", "IC50": "EC50", "Ratio": "GMR"}
    both_dict = {**units_dict, **terms_dict}

    for sample in inp_samples:
        html = sample["html"]
        labels = sample["accept"]
        task_hash = sample["_task_hash"]
        input_hash = sample["_input_hash"]
        units_dict_lower = {k.lower(): v for k, v in units_dict.items()}
        terms_dict_lower = {k.lower(): v for k, v in terms_dict.items()}
        both_dict_lower = {k.lower(): v for k, v in both_dict.items()}
        unit_pattern = '|'.join(sorted(re.escape(k) for k in units_dict))
        term_pattern = '|'.join(sorted(re.escape(k) for k in terms_dict))
        both_pattern = '|'.join(sorted(re.escape(k) for k in both_dict))

        def repl_units(match):
            matched = str(match.group(0)).lower()
            value = units_dict_lower[matched]
            return value

        def repl_terms(match):
            matched = str(match.group(0)).lower()
            value = terms_dict_lower[matched]
            return value

        def repl_both(match):
            matched = str(match.group(0)).lower()
            value = both_dict_lower[matched]
            return value

        units_html = re.sub(unit_pattern, repl_units, html, flags=re.IGNORECASE)
        terms_html = re.sub(term_pattern, repl_terms, html, flags=re.IGNORECASE)
        both_html = re.sub(both_pattern, repl_both, html, flags=re.IGNORECASE)

        augmented_samples.append(
            {"html": units_html, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})
        augmented_samples.append(
            {"html": terms_html, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})
        augmented_samples.append(
            {"html": both_html, "accept": labels, "_task_hash": task_hash, "_input_hash": input_hash})

    return augmented_samples


def augment_all(inp_samples: List[Dict]):
    """Agitates numbers by 10% up or down or replaces PK terms with synonyms,
    Please note input and task hashes will need to be reset at the end if this function is used """
    augmented_samples = []
    number_samples = augment_nums(inp_samples)
    syn_samples = augment_syns(inp_samples)
    augmented_samples.extend(number_samples)
    augmented_samples.extend(syn_samples)
    return augmented_samples


# my_test_data = [{"html": "<html><table><th> Mg mg min ml/kg ml/kg Cmax, k el", "accept": [1, 2, 3, 4, 5]},{"html": "<html><table><th> blah blah <\html>", "accept": [1, 2, 3, 4, 5]}]

def get_dataloaders(inp_data_dir: str, inp_tokenizer: str, max_len: int,
                    batch_size: int, val_batch_size: int, n_workers: int, remove_html: bool, baseline_only: bool,
                    aug_all: bool, aug_nums: bool, aug_syns: bool) -> \
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

    if aug_all:
        augmented_samples = augment_all(train_samples)
        train_samples.extend(augmented_samples)
    if aug_nums:
        augmented_samples = augment_nums(train_samples)
        train_samples.extend(augmented_samples)
    if aug_syns:
        augmented_samples = augment_syns(train_samples)
        train_samples.extend(augmented_samples)

    print(f"Total Length of Train Samples is: {len(train_samples)}")

    train_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                      "_input_hash": dic["_input_hash"]} for dic in train_samples]
    valid_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                      "_input_hash": dic["_input_hash"]} for dic in valid_samples]
    test_samples = [{"html": dic["html"], "accept": dic["accept"], "_task_hash": dic["_task_hash"],
                     "_input_hash": dic["_input_hash"]} for dic in test_samples]

    train_dataloader, train_dataset = make_dataloader(inp_samples=train_samples, batch_size=batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=True, sampler=False, n_workers=n_workers,
                                                      dataset_name='training', remove_html=remove_html,
                                                      baseline_only=baseline_only,
                                                      remove_5s=False)
    valid_dataloader, valid_dataset = make_dataloader(inp_samples=valid_samples, batch_size=val_batch_size,
                                                      inp_tokenizer=inp_tokenizer,
                                                      max_len=max_len, shuffle=False, sampler=False, n_workers=n_workers,
                                                      dataset_name='dev', remove_html=remove_html,
                                                      baseline_only=baseline_only,
                                                      remove_5s=False)
    test_dataloader, test_dataset = make_dataloader(inp_samples=test_samples, batch_size=val_batch_size,
                                                    inp_tokenizer=inp_tokenizer,
                                                    max_len=max_len, shuffle=False, sampler=False, n_workers=n_workers,
                                                    dataset_name='test', remove_html=remove_html,
                                                    baseline_only=baseline_only,
                                                    remove_5s=True)

    return train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset


