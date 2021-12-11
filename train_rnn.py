import torch
from tqdm import trange
import argparse
import random

from languages import Language
from utils import Tokenizer, get_data
from models import Tagger
from sampling import BalancedSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stop_threshold", type=int, default=2)
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_dev", type=int, default=1000)
    parser.add_argument("--train_length", type=int, default=100)
    parser.add_argument("--dev_length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    return parser.parse_args()

args = parse_args()
use_gpu = torch.cuda.is_available()
tokenizer = Tokenizer()
lang = Language.from_string(args.lang)
sampler = BalancedSampler(lang)
if lang is None:
    raise NotImplementedError("Non implemented language.")

random.seed(args.seed)
torch.random.manual_seed(args.seed)

print("Generating dataset...")
train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, args.n_train, args.train_length)
dev_tokens, dev_labels, dev_mask, dev_sents = get_data(sampler, lang, tokenizer, args.n_dev, args.dev_length)

print("Sample input")
print("tokens", train_tokens[3, :10])
print("labels", train_labels[3, :10])
print("mask  ", train_mask[3, :10].bool())
print("ntoken", tokenizer.n_tokens)

model = Tagger(tokenizer.n_tokens, 10, 100)
if use_gpu:
    model.cuda()
optim = torch.optim.AdamW(model.parameters())

best_acc = 0.
best_epoch = -1

for epoch in range(args.n_epochs):
    print(f"Starting epoch {epoch}...")
    perm = torch.randperm(len(train_tokens))
    train_tokens = train_tokens[perm, :]
    train_labels = train_labels[perm, :]
    train_mask = train_mask[perm, :]

    for batch_idx in trange(0, len(train_tokens) - args.batch_size, args.batch_size):
        optim.zero_grad()
        batch_tokens = train_tokens[batch_idx:batch_idx + args.batch_size]
        batch_labels = train_labels[batch_idx:batch_idx + args.batch_size]
        batch_mask = train_mask[batch_idx:batch_idx + args.batch_size]
        if use_gpu:
            batch_tokens = batch_tokens.cuda()
            batch_labels = batch_labels.cuda()
            batch_mask = batch_mask.cuda()
        output_dict = model(batch_tokens, batch_labels, batch_mask)
        loss = output_dict["loss"]
        loss.backward()
        optim.step()

    with torch.no_grad():
        if use_gpu:
            dev_tokens = dev_tokens.cuda()
            dev_labels = dev_labels.cuda()
            dev_mask = dev_mask.cuda()
        dev_output_dict = model(dev_tokens, dev_labels, dev_mask)
        acc = dev_output_dict["accuracy"]
        print("Dev acc:", acc.item())

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

    if acc >= best_acc:
        path = "models/best" + str(args.lang) + ".th"
        print(f"Best model! Saved {path}")
        torch.save(model.state_dict(), path)

    if epoch - best_epoch > args.stop_threshold:
        print("Stopped early!")
        break
