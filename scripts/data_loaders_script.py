from data_loaders.table_data_loaders import get_dataloaders
import json

with open("../config/config_tableclass_FFNN.json") as config:
    cf = json.load(config)

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(
    inp_data_dir="../data/train-test-val",
    inp_tokenizer="../tokenizers/tokenizerPKtablesSpecialTokens5000.json",
    max_len=2800, batch_size=cf["batch_size"], val_batch_size=cf["val_batch_size"],
    n_workers=cf["n_workers"], remove_html=True, baseline_only=False,
    aug_all=False, aug_nums=False, aug_syns=False,
    aug_both=False, sampler=False, sections=False, multi_hot=False)

a= 1
