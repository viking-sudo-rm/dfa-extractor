from typing import Dict
import argparse
import os
import pickle
import torch
import numpy as np
from collections import defaultdict

from sklearn.cluster import KMeans

from languages import Language
from pythomata_wrapper import to_pythomata_dfa
from utils import Tokenizer, get_data
from sampling import BalancedSampler, TestSampler
from models import Tagger
from automaton import Dfa
from extract_dfa import score_whole_words


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom1")
    parser.add_argument("--eval", choices=["preds", "labels"], default="preds")
    parser.add_argument("--n_train", type=int, default=200)
    parser.add_argument("--len_train", type=int, default=10)
    parser.add_argument("--n_clusters", type=int, default=20)
    return parser.parse_args()


class KmeansExtractor:
    def __init__(self, train_tokens, train_labels, train_mask, train_results, tokenizer, n_clusters):
        self.train_tokens = train_tokens
        self.train_labels = train_labels
        self.train_mask = train_mask
        self.train_results = train_results
        self.tokenizer = tokenizer
        self.n_clusters = n_clusters

    def _cluster(self):
        """Do KMeans to create a list of clustered states."""
        self.raw_states = train_results["states"] * train_mask.unsqueeze(dim=-1)
        states = [state.numpy() for state in self.raw_states.flatten(end_dim=1) if not (state == 0).all()]
        self.kmeans = KMeans(n_clusters=self.n_clusters)
        clusters = self.kmeans.fit_predict(states)
        self.idxs_by_cluster = [[idx for idx, c_ in enumerate(clusters) if c_ == c] for c in range(self.n_clusters)]

    def _get_init_state(self):
        init = self.raw_states[:, 0, :]
        init_clusters = list(self.kmeans.predict(init))
        return self._majority_vote(init_clusters)

    def _get_accepting_mask(self):
        """Compute whether each cluster accepts or rejects by voting."""
        labels = self.train_labels.flatten()
        accept_probs = np.asarray([labels[idxs].float().mean() for idxs in self.idxs_by_cluster])
        return accept_probs > .5
    
    def _get_nfa_transitions(self):
        nfa_transitions: Dict[int, str] = defaultdict(list)
        for i, sentence in enumerate(self.train_tokens):
            last_cluster = None
            for j, tok_idx in enumerate(sentence):
                token = self.tokenizer.index_to_token[tok_idx.item()]
                if token == "<pad>":
                    break
                state = self.raw_states[i, j].numpy().reshape(1, -1)
                cluster = self.kmeans.predict(state).item()
                if last_cluster is not None:
                    nfa_transitions[last_cluster, token].append(cluster)
                last_cluster = cluster
        return nfa_transitions
    
    def _get_dfa_transitions(self):
        nfa_transitions = self._get_nfa_transitions()
        return {tup: self._majority_vote(values) for tup, values in nfa_transitions.items()}
    
    def get_dfa(self, name: str):
        self._cluster()
        init_state = self._get_init_state()
        accepting = self._get_accepting_mask()
        transitions = self._get_dfa_transitions()
        new_trans = [(k1, k2, v) for (k1, k2), v in transitions.items()]
        return Dfa(name, list(enumerate(accepting)), new_trans, init_state=init_state)
    
    @staticmethod
    def _majority_vote(values: list):
        """Could make this faster probably."""
        return max(set(values), key=values.count)


args = parse_args()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lang_name = args.lang
lang = Language.from_string(lang_name.split("-")[0])  # Remove any clarifying suffixes
lang_dir = os.path.join("models", lang_name)
token_path = os.path.join(lang_dir, "tokenizer.pkl")
with open(token_path, "rb") as fh:
    tokenizer: Tokenizer = pickle.load(fh)

sampler = BalancedSampler(lang)
dev_sampler = TestSampler(lang)
train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, args.n_train, args.len_train)
val_tokens, val_labels, val_mask, val_sents = get_data(sampler, lang, tokenizer, 20, 2 * args.len_train)
dev_tokens, _dev_labels, dev_mask, dev_sents = get_data(dev_sampler, lang, tokenizer, 1000, 50)
dev_labels = [_dev_labels[i][dev_mask[i]][-1] for i in range(len(_dev_labels))]

path = os.path.join(lang_dir, "best.th")
trained_model = Tagger(tokenizer.n_tokens, 10, 100)
trained_model.load_state_dict(torch.load(path, map_location=device))
with torch.no_grad():
    train_results = trained_model(train_tokens, train_labels, train_mask)
    val_results = trained_model(val_tokens, val_labels, val_mask)
    dev_results = trained_model(dev_tokens, _dev_labels, dev_mask)
train_preds = train_results["predictions"]
val_preds = val_results["predictions"]
_dev_preds = dev_results["predictions"]
dev_preds = [_dev_preds[i][dev_mask[i]][-1] for i in range(len(_dev_preds))] # valid for TestSampler

if (args.eval == "preds"):
    train_gold = train_preds
    val_gold = val_preds
    dev_gold = dev_preds
elif (args.eval == "labels"):
    train_gold = train_labels
    val_gold = val_labels
    dev_gold = dev_labels
else:
    raise ValueError("Choose --eval between predictions `preds` and labels `labels`.")

with torch.no_grad():
    extractor = KmeansExtractor(train_tokens, train_labels, train_mask, train_results, tokenizer, args.n_clusters)
    dfa = extractor.get_dfa(name=lang_name + "-kmeans")

dev_gold = [bool(x) for x in dev_gold]
dev_acc = score_whole_words(dfa, dev_sents, dev_gold)
print(f"{lang_name} Acc", dev_acc)
