import json
import pytorch_lightning as pl
import torch
import gc
from pytorch_lightning.core.decorators import auto_move_data
from transformers import PreTrainedTokenizerFast
from data_loaders.table_data_loaders import get_dataloaders
from data_loaders.table_data_loaders import NeuralNet
from tableclass_engine import get_tensorboard_logger
import typer
from pathlib import Path
from datetime import datetime
import os


def main(
        data_dir: Path = typer.Option(default="../data/train-test-val/",
                                      help="Data directory expecting a training,dev and test.jsonl files"),
        output_dir: Path = typer.Option(default="../data/outputs",
                                        help="Output directory"),
        model_config_file: Path = typer.Option(default="../config/config.json"),

):
    gc.collect()
    torch.cuda.empty_cache()
    with open(model_config_file) as cf:
        config = json.load(cf)

    config['data_dir'] = str(data_dir)
    config['output_dir'] = str(output_dir)
    current_time = datetime.now().strftime('%d-%m-%Y-%H-%M-%S')
    out_config_path = os.path.join(config['output_dir'], "model_saves",
                                   f"config-{config['run_name']}-{current_time}.json")
    if not os.path.exists(os.path.join(config['output_dir'], "model_saves")):
        os.makedirs(os.path.join(config['output_dir'], "model_saves"))

    with open(out_config_path, 'w') as fp:
        json.dump(config, fp, indent=3)

    # ============ Seed everything =============== #
    pl.seed_everything(config['seed'])

    # ============ Load tokenizer and make knowledge base =========== #
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=config["base_tokenizer"])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    # ============ Get data loaders and datasets =============== #
    train_dataloader, train_dataset, val_dataloader, val_dataset, test_dataloader, test_dataset = get_dataloaders(
        inp_data_dir=config['data_dir'],
        inp_tokenizer=tokenizer,
        max_len=config['max_length'],
        batch_size=config['batch_size'],
        val_batch_size=config['val_batch_size'],
        n_workers=config['n_workers_dataloader']
    )

    total_training_steps = len(train_dataloader) * config["max_epochs"]

    # ============ Get model =============== #
    NeuralNet.forward = auto_move_data(NeuralNet.forward)  # auto move data to the correct device
    model = NeuralNet(input_size=input_size, num_classes=num_classes, embeds_size=embeds_size, vocab_size=vocab_size,
                      padding_idx=padding_idx, hidden_size=hidden_size)

    # ============ Define trainer =============== #

    tensorboard_logger = get_tensorboard_logger(log_dir=os.path.join(config['output_dir'], "logs"),
                                                run_name=config['run_name'])

    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_f1_strict',
                                                       dirpath=os.path.join(config['output_dir'], 'checkpoints'),
                                                       filename=config['run_name'] + '-{epoch:04d}-{val_f1_strict:.2f}',
                                                       save_top_k=1,
                                                       mode='max',

                                                       )

    if config['gpus']:
        gpus = torch.cuda.device_count()
    else:
        gpus = 0

    trainer = pl.Trainer(
        gpus=gpus, max_epochs=config['max_epochs'], logger=tensorboard_logger,
        callbacks=[checkpoint_callback],
        # val_check_interval=0.25,
        # check_val_every_n_epoch =1,
        # check_val_every_n_epoch=1,
        deterministic=True, log_every_n_steps=1,
        # gradient_clip_val=config["gradient_clip_val"],
        limit_train_batches=2, limit_val_batches=2  # JUST FOR QUICK DEBUGGING!
    )

    if config['early_stopping']:
        early_stop = pl.callbacks.EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'],
                                                strict=False, verbose=True, mode='min')
        trainer.callbacks += [early_stop]

    # ============ Train model =============== #
    trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)


if __name__ == '__main__':
    typer.run(main)
