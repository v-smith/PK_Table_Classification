from data_loaders.table_data_loaders import get_dataloaders

train_dataloader, train_dataset, valid_dataloader, valid_dataset, test_dataloader, test_dataset = get_dataloaders(inp_data_dir="../data/train-test-val",
                                                                       inp_tokenizer="../tokenizers/tokenizeralltablesNotestNohtml5000.json",
                                                                       max_len=4000, batch_size=50, val_batch_size=100,
                                                                       n_workers=0)
