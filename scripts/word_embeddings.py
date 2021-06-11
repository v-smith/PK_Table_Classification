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


my_list = ["<html><table><th> blah", "<html><table><th> blah <\html>", "<html><table><th> blah blah <\html>"]
tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
encodings = tokenizer(my_list, padding= True, truncation= True, max_length=10)
print(encodings)
