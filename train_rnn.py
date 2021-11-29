import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import trange
import argparse

from abstar import AbstarGenerator
from utils import sequence_cross_entropy_with_logits


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--stop_threshold", type=int, default=2)
    return parser.parse_args()


class Tokenizer:
    def __init__(self):
        self.next_idx = 0
        self.token_to_index = {}
        self.index_to_token = {}
        self.to_index("<pad>")
        self.to_index("<unk>")
    
    def to_index(self, token, add=True):
        if token not in self.token_to_index:
            if add:
                self.token_to_index[token] = self.next_idx
                self.index_to_token[self.next_idx] = token
                self.next_idx += 1
            else:
                return self.token_to_index["<unk>"]
        return self.token_to_index[token]
    
    def tokenize(self, sentence, add=True):
        return [self.to_index(token, add=add) for token in sentence]
    
    @property
    def n_tokens(self):
        return self.next_idx


class LanguageModel(torch.nn.Module):
    def __init__(self, n_tokens, embed_dim, rnn_dim):
        super().__init__()
        self.embed = torch.nn.Embedding(n_tokens, embed_dim)
        self.rnn = torch.nn.RNN(embed_dim, rnn_dim, batch_first=True)
        self.output = torch.nn.Linear(rnn_dim, n_tokens)

        self.criterion = torch.nn.CrossEntropyLoss()
    
    def forward(self, token_ids, mask):
        embeddings = self.embed(token_ids[:, :-1])
        states, _ = self.rnn(embeddings)
        logits = self.output(states)
        labels = token_ids[:, 1:].contiguous()
        label_mask = mask[:, :-1].contiguous()
        loss = sequence_cross_entropy_with_logits(logits, labels, label_mask)
        return {
            "states": states,
            "predictions": logits.argmax(dim=-1),
            "loss": loss,
        }


args = parse_args()
use_gpu = torch.cuda.is_available()
tokenizer = Tokenizer()
lang = AbstarGenerator()
train = pad_sequence([torch.tensor(tokenizer.tokenize(sent)) for sent in lang.generate(1, 1000)], batch_first=True)
dev = pad_sequence([torch.tensor(tokenizer.tokenize(sent, add=False)) for sent in lang.generate(1001, 1100)], batch_first=True)
train_mask = (train != 0)
dev_mask = (dev != 0)

print("Sample input")
print(train[3, :10])
print(train_mask[3, :10])

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
        print("Best model! Saved models/best.th")
        torch.save(model.state_dict(), "models/best.th")
    
    if epoch - best_epoch > args.stop_threshold:
        print("Stopped early!")
        break
