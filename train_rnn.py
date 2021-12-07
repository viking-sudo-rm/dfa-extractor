import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
import argparse

from languages import *
from utils import sequence_cross_entropy_with_logits, LanguageModel, Tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stop_threshold", type=int, default=2)
    parser.add_argument("--lang", type=str, default="Tom2")
    return parser.parse_args()

args = parse_args()
use_gpu = torch.cuda.is_available()
tokenizer = Tokenizer()
if (args.lang == "Tom1"):
    lang = Tomita1()
elif (args.lang == "Tom2"):
    lang = Tomita2()
elif (args.lang == "Tom3"):
    lang = Tomita3()
elif (args.lang == "Tom4"):
    lang = Tomita4()
elif (args.lang == "Tom5"):
    lang = Tomita5()
elif (args.lang == "Tom6"):
    lang = Tomita6()
elif (args.lang == "Tom7"):
    lang = Tomita7()
elif (args.lang == "abbastar"):
    lang = AbbastarGenerator()
else:
    raise NotImplementedError("Non implemented language.")

# increase dataset for Tomita5
train = pad_sequence([torch.tensor(tokenizer.tokenize(sent)) for sent in lang.generate(0, 1000)], batch_first=True)
dev = pad_sequence([torch.tensor(tokenizer.tokenize(sent, add=False)) for sent in lang.generate(1001, 1100)], batch_first=True)
train_mask = (train != 0)
dev_mask = (dev != 0)

print("Sample input")
print(train[3, :10])
print(train_mask[3, :10])
print(tokenizer.n_tokens)

model = LanguageModel(tokenizer.n_tokens, 10, 100)
if use_gpu:
    model.cuda()
optim = torch.optim.AdamW(model.parameters())

best_acc = 0.
best_epoch = -1

for epoch in range(args.n_epochs):
    print(f"Starting epoch {epoch}...")
    perm = torch.randperm(len(train))
    train = train[perm, :]

    for batch_idx in trange(0, len(train) - args.batch_size, args.batch_size):
        optim.zero_grad()
        train_batch = train[batch_idx:batch_idx + args.batch_size]
        train_mask_batch = train_mask[batch_idx:batch_idx + args.batch_size]
        if use_gpu:
            train_batch = train_batch.cuda()
            train_mask_batch = train_mask_batch.cuda()
        output_dict = model(train_batch, train_mask_batch)
        loss = output_dict["loss"]
        loss.backward()
        optim.step()

    with torch.no_grad():
        if use_gpu:
            dev = dev.cuda()
            dev_mask = dev_mask.cuda()
        dev_output_dict = model(dev, dev_mask)
        predictions = dev_output_dict["predictions"]
        shift_mask = dev_mask[:, :-1]
        acc = ((predictions == dev[:, 1:]) * shift_mask).sum() / shift_mask.sum()
        print("Dev acc:", acc.item())

    if acc > best_acc:
        best_acc = acc
        best_epoch = epoch
        print("Best model! Saved models/best'--lang'.th")
        torch.save(model.state_dict(), "models/best" + str(args.lang) + ".th")

    if epoch - best_epoch > args.stop_threshold:
        print("Stopped early!")
        break
