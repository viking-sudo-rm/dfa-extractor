import torch
from tqdm import trange
import argparse
import random
import os
import pickle
from collections import defaultdict

from languages import Language
from utils import Tokenizer, get_data, get_norm
from models import Tagger
from sampling import BalancedSampler

def add_noise(labels, percentage):
    noise_part = int(percentage * len(labels))
    temp_labels = ~(labels[:noise_part].bool())
    labels[:noise_part] = temp_labels.int()
    return labels

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--stop_threshold", type=int, default=2)
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--n_train", type=int, default=100000)
    parser.add_argument("--n_dev", type=int, default=1000)
    parser.add_argument("--train_length", type=int, default=100)
    parser.add_argument("--dev_length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--only_tokenize", action="store_true")
    parser.add_argument("--save_name", type=str, default=None)
    parser.add_argument("--save_all", action="store_true")
    parser.add_argument("--noise_percentage", type=float, default=0.)
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

print("clean labels", train_labels[:2, :3])
train_labels = add_noise(train_labels, args.noise_percentage)
print("noisy_labels", train_labels[:2, :3])

print("Sample input")
print("tokens", train_tokens[3, :10])
print("labels", train_labels[3, :10])
print("mask  ", train_mask[3, :10].bool())
print("ntoken", tokenizer.n_tokens)

save_name = args.save_name if args.save_name is not None else args.lang
model_dir = os.path.join("models", save_name)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
token_path = os.path.join(model_dir, "tokenizer.pkl")
with open(token_path, "wb") as fh:
    pickle.dump(tokenizer, fh)
if args.only_tokenize:
    quit()

model = Tagger(tokenizer.n_tokens, 10, 100)
if use_gpu:
    print(f"Using CUDA device {args.device}")
    model.cuda(args.device)
if (args.lr != None):
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
else:
    optim = torch.optim.AdamW(model.parameters())

best_acc = 0.
best_epoch = -1

saved_epochs = []
metrics = defaultdict(list)

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
            batch_tokens = batch_tokens.cuda(args.device)
            batch_labels = batch_labels.cuda(args.device)
            batch_mask = batch_mask.cuda(args.device)
        output_dict = model(batch_tokens, batch_labels, batch_mask)
        loss = output_dict["loss"]
        loss.backward()
        optim.step()

    print("=== Train metrics ===")
    print("\tacc =", output_dict["accuracy"].item())
    print("\tloss =", output_dict["loss"].item())

    with torch.no_grad():
        if use_gpu:
            dev_tokens = dev_tokens.cuda(args.device)
            dev_labels = dev_labels.cuda(args.device)
            dev_mask = dev_mask.cuda(args.device)
        dev_output_dict = model(dev_tokens, dev_labels, dev_mask)
        metrics["loss"] = dev_output_dict["loss"].item()
        metrics["accuracy"] = dev_output_dict["accuracy"].item()
        metrics["norm"] = get_norm(model.rnn).item()
        print("=== Dev metrics ===")
        print("\tacc =", metrics["accuracy"])
        print("\tloss =", metrics["loss"])
        print("\tnorm =", metrics["norm"])
        acc = metrics["accuracy"]

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch

    if args.save_all or acc >= best_acc:
        best_path = os.path.join(model_dir, "best.th")
        torch.save(model.state_dict(), best_path)
        epoch_path = os.path.join(model_dir, f"epoch{epoch}.th")
        torch.save(model.state_dict(), epoch_path)
        saved_epochs.append(epoch)
        print(f"Saved checkpoint!")

    if epoch - best_epoch > args.stop_threshold:
        print("Stopped early!")
        break

print("Saving metrics dict...")
metrics_path = os.path.join(model_dir, "metrics.pkl")
with open(metrics_path, "wb") as fh:
    pickle.dump(metrics, fh)

print("Cleaning up sub-optimal checkpoints...")
if not args.save_all:
    for epoch in saved_epochs:
        if epoch < best_epoch:
            epoch_path = os.path.join(model_dir, f"epoch{epoch}.th")
            os.remove(epoch_path)
