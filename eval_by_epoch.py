from copy import deepcopy
import os
import re
import argparse
import torch
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import tqdm
import random

from languages import Language
from utils import Tokenizer, get_data
from models import Tagger
from sampling import BalancedSampler, TestSampler
from extract_dfa import build_dfa_from_dict, cosine_merging, score_all_prefixes, score_whole_words
from pythomata_wrapper import to_pythomata_dfa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--n_train", type=int, default=100)
    parser.add_argument("--eval", choices=["preds", "labels"], default="preds")
    parser.add_argument("--sim_threshold", type=float, default=.99)
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--n_seeds", type=int, default=1)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    torch.random.manual_seed(seed)


def get_metrics(args, lang_name: str):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lang = Language.from_string(lang_name)
    tokenizer = Tokenizer()
    sampler = BalancedSampler(lang)
    dev_sampler = TestSampler(lang)
    lang_dir = os.path.join("models", lang_name)
    ckpts = [s for s in os.listdir(lang_dir) if re.match(r"epoch[0-9]+\.th", s)]
    ckpt_ids = [int(re.match(r"epoch([0-9]+)\.th", s)[1]) for s in ckpts]
    pairs = list(zip(ckpts, ckpt_ids))
    pairs.sort(key=lambda pair: pair[1])
    ckpts, ckpt_ids = zip(*pairs)
    print("Got checkpoints:", ckpt_ids)

    # Default dict of default dict of lists
    metrics = defaultdict(lambda: defaultdict(list))
    for ckpt in tqdm.tqdm(ckpts):
        path = os.path.join(lang_dir, ckpt)
        for seed in range(args.n_seeds):
            set_seed(seed)
            train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, args.n_train, 10)
            dev_tokens, _dev_labels, dev_mask, dev_sents = get_data(dev_sampler, lang, tokenizer, 1000, 50)
            dev_labels = [_dev_labels[i][dev_mask[i]][-1] for i in range(len(_dev_labels))]
            trained_model = Tagger(tokenizer.n_tokens, 10, 100)
            trained_model.load_state_dict(torch.load(path, map_location=device))
            with torch.no_grad():
                train_results = trained_model(train_tokens, train_labels, train_mask)
                dev_results = trained_model(dev_tokens, _dev_labels, dev_mask)
            train_preds = train_results["predictions"]
            _dev_preds = dev_results["predictions"]
            dev_preds = [_dev_preds[i][dev_mask[i]][-1] for i in range(len(_dev_preds))] # valid for TestSampler

            if (args.eval == "preds"):
                train_gold = train_preds
                dev_gold = dev_preds
            elif (args.eval == "labels"):
                train_gold = train_labels
                dev_gold = dev_labels
            else:
                raise ValueError("Choose --eval between predictions `preds` and labels `labels`.")

            redundant_dfa = build_dfa_from_dict(id=args.lang, dict=train_sents, labels=train_gold) # build the trie based on train_gold
            assert(score_all_prefixes(redundant_dfa, train_sents, train_gold) == 100.)
            representations = train_results["states"]
            idx = [redundant_dfa.return_states(sent) for sent in train_sents]
            n_states = len(redundant_dfa.table.keys())
            states = torch.empty((n_states, 100))
            states_mask = torch.empty((n_states), dtype=torch.long)
            for i, _r in enumerate(representations):
                states[idx[i]] = _r[train_mask[i]]
                states_mask[idx[i]] = train_labels[i][train_mask[i]]

            # Merge states
            init_dfa = deepcopy(redundant_dfa)
            merge_dfa = cosine_merging(redundant_dfa, states, states_mask, threshold=args.sim_threshold)
            merge_pdfa = to_pythomata_dfa(merge_dfa)
            min_pdfa = merge_pdfa.minimize().trim()

            # Evaluate performance
            init_train_acc = score_all_prefixes(init_dfa, train_sents, train_gold)
            merge_train_acc = score_all_prefixes(merge_dfa, train_sents, train_gold)
            init_dev_acc = score_whole_words(init_dfa, dev_sents, dev_gold)
            merge_dev_acc = score_whole_words(merge_dfa, dev_sents, dev_gold) # valid for TestSampler

            metrics["init_train_acc"][seed].append(init_train_acc.item())
            metrics["merge_train_acc"][seed].append(merge_train_acc.item())
            metrics["init_dev_acc"][seed].append(init_dev_acc.item())
            metrics["merge_dev_acc"][seed].append(merge_dev_acc.item())

            metrics["merge_n_states"][seed].append(len(merge_pdfa.states))
            metrics["min_n_states"][seed].append(len(min_pdfa.states))

    metrics = {k: dict(d) for k, d in metrics.items()}
    return ckpt_ids, metrics

args = parse_args()
tomita_metrics_file = f"cached/tomita_metrics-{args.n_train}.pkl"
ckpt_ids_file = f"cached/ckpt_ids.pkl"

if args.load:
    with open(tomita_metrics_file, "rb") as fh:
        tomita_metrics = pickle.load(fh)
    with open(ckpt_ids_file, "rb") as fh:
        ckpt_ids = pickle.load(fh)
else:
    # Can modify this line to only do some of the Tomita languages.
    lang_names = [f"Tom{i}" for i in range(1, 8)]
    tomita_metrics = {}
    ckpt_ids = {}
    for idx in range(1, 8):
        lang_name = f"Tom{idx}"
        print(f"Starting {lang_name}...")
        ckpt_ids_, metrics = get_metrics(args, lang_name)
        ckpt_ids[lang_name] = ckpt_ids_
        tomita_metrics[lang_name] = metrics
    with open(tomita_metrics_file, "wb") as fh:
        pickle.dump(tomita_metrics, fh)
    with open(ckpt_ids_file, "wb") as fh:
        pickle.dump(ckpt_ids, fh)

nice_names = {
    "init_train_acc": "dev acc of prefix tree",
    "init_dev_acc": "train acc of prefix tree",
    "merge_dev_acc": "dev acc of merged DFA",
    "merge_train_acc": "train acc of merged DFA",
    "merge_n_states": "#states in merged DFA",
    "min_n_states": "#states in minimized DFA",
}

colors = ["blue", "green", "red", "cyan", "magenta", "yellow", "black"]

by_epoch_dir = "images/by-epoch"
if not os.path.isdir(by_epoch_dir):
    os.makedirs(by_epoch_dir)

for name, nice_name in nice_names.items():
    plt.figure()
    plt.xlabel("#epochs")
    plt.ylabel(nice_name)
    plt.tight_layout()
    for color, (lang_name, metrics) in zip(colors, tomita_metrics.items()):
        ids = ckpt_ids[lang_name]
        values = torch.tensor(list(metrics[name].values()), dtype=torch.float)
        mids = torch.quantile(values, q=.5, dim=0)
        lows = torch.quantile(values, q=.25, dim=0)
        highs = torch.quantile(values, q=.75, dim=0)
        plt.fill_between(ids, lows, highs, alpha=.2, color=color)
        plt.plot(ids, mids, label=lang_name, color=color)
    plt.legend()
    filename = f"{name}-{args.n_train}.pdf"
    plt.savefig(os.path.join(by_epoch_dir, filename))
