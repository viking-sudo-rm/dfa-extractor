import torch
from utils import sequence_cross_entropy_with_logits

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

class Tagger(torch.nn.Module):
    def __init__(self, n_tokens, embed_dim, rnn_dim, n_labels=2):
        super().__init__()
        self.embed = torch.nn.Embedding(n_tokens, embed_dim)
        self.rnn = torch.nn.RNN(embed_dim, rnn_dim, batch_first=True)
        self.output = torch.nn.Linear(rnn_dim, n_labels)
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, token_ids, labels, mask):
        embeddings = self.embed(token_ids)
        states, _ = self.rnn(embeddings)
        logits = self.output(states)
        loss = sequence_cross_entropy_with_logits(logits, labels, mask)
        predictions = logits.argmax(dim=-1)
        acc = ((predictions == labels) * mask).sum().float() / mask.sum()
        return {
            "states": states,
            "predictions": predictions,
            "accuracy": acc,
            "loss": loss,
        }
