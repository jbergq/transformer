from functools import partial

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

import torchtext.transforms as T
import torchtext.functional as F
from torchtext.datasets import IMDB

from src.model.transformer import Transformer
from src.data.preprocess import PreProcess
from src.utils import ExperimentLogger, iter_print, epoch_print, load_config


def train(model, dataloader, optimizer, criterion, epoch):
    train_losses = []

    model.train()

    for i, batch in enumerate(dataloader):
        src, tgt = batch["source"], batch["target"]

        out = model(src, tgt)

        out_reshape = out.contiguous().view(-1, out.shape[-1])
        tgt_reshape = tgt.contiguous().view(-1)

        loss = criterion(out_reshape, tgt_reshape)
        loss.backward()
        optimizer.step()

        iter_print(epoch, i, loss.item())
        train_losses.append(loss.item())

        if i == 10:
            break

    return torch.tensor(train_losses)


def validate(model, dataloader, criterion, epoch):
    val_losses = []

    model.eval()

    for i, batch in enumerate(dataloader):
        src, tgt = batch["source"], batch["target"]

        out = model(src, tgt)

        out_reshape = out.contiguous().view(-1, out.shape[-1])
        tgt_reshape = tgt.contiguous().view(-1)

        loss = criterion(out_reshape, tgt_reshape)

        iter_print(epoch, i, loss.item())
        val_losses.append(loss.item())

        if i == 10:
            break

    return torch.tensor(val_losses)


def run():
    cfg = load_config("config.yaml")

    train_dp, val_dp = IMDB(split=("train", "test"))

    encoder_json_path = "https://download.pytorch.org/models/text/gpt2_bpe_encoder.json"
    vocab_bpe_path = "https://download.pytorch.org/models/text/gpt2_bpe_vocab.bpe"

    tokenizer = T.GPT2BPETokenizer(encoder_json_path, vocab_bpe_path)
    vocab_path = "https://download.pytorch.org/models/text/roberta.vocab.pt"
    vocab = load_state_dict_from_url(vocab_path)

    transform = PreProcess(tokenizer, vocab, cfg.sequence_length)

    train_dp = train_dp.batch(cfg.batch_size).rows2columnar(["label", "text"])
    train_dp = train_dp.map(transform)
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="source")
    train_dp = train_dp.map(partial(F.to_tensor, padding_value=1), input_col="target")

    val_dp = val_dp.batch(cfg.batch_size).rows2columnar(["label", "text"])
    val_dp = val_dp.map(transform)
    val_dp = val_dp.map(partial(F.to_tensor, padding_value=1), input_col="source")
    val_dp = val_dp.map(partial(F.to_tensor, padding_value=1), input_col="target")

    train_dataloader = DataLoader(train_dp, batch_size=None)
    val_dataloader = DataLoader(val_dp, batch_size=None)

    model = Transformer(len(vocab), 100, 128, 512)
    optimizer = Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, eps=5e-9
    )
    criterion = nn.CrossEntropyLoss(ignore_index=1)

    logger = ExperimentLogger(log_dir="./experiments")

    for epoch in range(cfg.num_epochs):
        train_losses = train(model, train_dataloader, optimizer, criterion, epoch)
        val_losses = validate(model, val_dataloader, criterion, epoch)

        logger.log_params(
            {
                "train_loss": train_losses.mean().item(),
                "val_loss": val_losses.mean().item(),
            }
        )
        epoch_print(epoch, val_losses)


if __name__ == "__main__":
    run()
