import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from typing import Dict, List, Iterable
from transformers import PreTrainedTokenizerFast
from data_loaders.table_data_loaders import extract_all_ids_withoutpad, preprocess_htmls
from collections import Counter
import jsonlines
import itertools
from itertools import islice
import operator
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

matplotlib.style.use('ggplot')

with jsonlines.open("../data/train-test-val/test-corrected.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

labels= [x["accept"] for x in json_list]
print(labels)


def plot_per_label(json_list: List[Dict], label: int):
    # plot the tokens per label or just the words to get an idea (without html)
    plot_list = []
    for item in json_list:
        labels = item["accept"]
        for x in labels:
            if x == label:
                plot_list.append(item)

    plot_htmls = [x["html"] for x in plot_list]
    plot_htmls = preprocess_htmls(plot_htmls, remove_html=True)
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="../tokenizers/tokenizerPKtablesSpecialTokens5000.json")
    table_encodings = tokenizer(plot_htmls)
    table_ids = extract_all_ids_withoutpad(table_encodings["input_ids"])
    tokens = [tokenizer.convert_ids_to_tokens(x) for x in table_ids]
    all_tokens = list(itertools.chain.from_iterable(tokens))
    counts = Counter(all_tokens)
    keys_to_be_deleted = ['▁', '_', "#n#", "-", ",", ')', '▁–', '‐', '▁%', '▁−']
    final_counts = {}
    for key, value in counts.items():
        if key not in keys_to_be_deleted:
            final_counts[key] = value
    #sorted_tuples = sorted(final_counts.items(), key=lambda item: item[1])
    #sorted_counts = {k: v for k, v in sorted_tuples}
    sorted_counts = dict(sorted(final_counts.items(), key=operator.itemgetter(1), reverse=True))
    print(sorted_counts)
    return sorted_counts

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))

def plot_label_stats(counts: Dict, label: int):
    """Plots total tokens per table"""
    first_100 = {k: counts[k] for k in list(counts)[:100]}
    print(first_100)
    plt.figure(figsize=(20, 10))
    plt.bar(range(len(first_100)), list(first_100.values()), align='center')
    plt.xticks(range(len(first_100)), list(first_100.keys()), rotation=90)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.title(f"Token frequencies for label {label}")
    plt.xlabel("tokens")
    plt.ylabel("frequency")
    plt.tight_layout
    plt.show()
    plt.close()

def plot_wordcloud(counts):
    keys = counts.keys()
    joined = ' '.join(keys)
    wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(joined)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    #wordcloud.to_file("img/first_review.png") #save
    plt.show()

counts = plot_per_label(json_list, 0)
plot_label_stats(counts, 0)
plot_wordcloud(counts)
a=1
