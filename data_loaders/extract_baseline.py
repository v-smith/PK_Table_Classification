# imports
import bs4 as bs
import jsonlines
import re
from typing import Dict, List, Iterable

def title_header_cols(inp_samples: List[str]):
    """Extracts just title, first column and first row contents from table"""
    processed_htmls = []
    for html in inp_samples:
        title = re.findall("\<h4>(.*?)\</h4>", html)
        soup = bs.BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        row_names = finall_rownames(table)
        col_names = findall_colnames(table)
        final_html = join_info(title, col_names, row_names)
        processed_htmls.append(final_html)

    return processed_htmls


# read in data
with jsonlines.open("../data/train-test-val/test.jsonl") as reader:
    json_list = []
    for obj in reader:
        json_list.append(obj)

html = json_list[["text"] == "Pmid= 6640488:Table= Table 1"]["html"]

# extract title= contents of <h4> </h4> and remove rows
title = re.findall("\<h4>(.*?)\</h4>", html)

# extract header
soup = bs.BeautifulSoup(html, 'lxml')
table = soup.find('table')
table_rows = table.find_all('tr')


# find all column names (header row)
def findall_colnames(table):
    column_names = []
    for row in table.find_all('tr'):
        th_tags = row.find_all('th')
        if len(th_tags) > 0 and len(column_names) == 0:
            for th in th_tags:
                column_names.append(th.get_text())
    return column_names


# find all row names (first col)
def findall_rownames(table):
    table_rows = table.find_all('tr')
    row_names = []
    for tr in table_rows:
        td = tr.find_all('td')
        row = [i.text for i in td]
        if row:
            row_names.append(row[0])
    return row_names


def join_info(title, column_names, row_names):
    final_list = []
    final_list.extend(title)
    final_list.extend(column_names)
    final_list.extend(row_names)
    final_str = ' '.join(final_list)
    return final_str
