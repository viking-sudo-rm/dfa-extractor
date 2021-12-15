import torch
import argparse
from copy import deepcopy
import random
from collections import defaultdict
import os
import pickle
import numpy as np

from automaton import Dfa, equiv
from pythomata_wrapper import to_pythomata_dfa
from trie import Trie
from languages import Language
from create_plot import create_plot
from models import Tagger
from utils import get_data, Tokenizer
from sampling import BalancedSampler, TestSampler
from pythomata_wrapper import to_pythomata_nfa, to_pythomata_dfa, from_pythomata_dfa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--train_length", type=int, default=10)
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=6)
    parser.add_argument("--sim_threshold", type=float, default=.99)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fst", dest='fst', action='store_true')
    parser.set_defaults(fst=False)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.set_defaults(min=False)
    parser.add_argument('--epoch', type=str, default="best")
    parser.add_argument('--eval', type=str, default="preds")
    parser.add_argument("--no_state_count", action="store_true")
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--find_threshold", action="store_true")
    return parser.parse_args()


def score_whole_words(dfa, dataset, labels):
    acc = 0
    for word, y in zip(dataset, labels):
        acc += (dfa.accept(word) == y)
    return (acc / len(dataset) * 100)

def score_all_prefixes(dfa, dataset, labels):
    # Evaluate the performance of the extracted DFA on the dataset
    count, acc = 0, 0
    for i, word in enumerate(dataset):
        cur = []
        # cur = ''
        for j, char in enumerate(word):
            acc += (dfa.accept(cur) == labels[i][j])
            # cur += char
            cur.append(char)
            count += 1
        # if (cur != ''):
        if cur:
            acc += (dfa.accept(cur) == labels[i][j+1]) # complete word
            count += 1
        else:
            acc += (dfa.accept(cur) == labels[i][0]) # empty string corner case
            count += 1
    return (acc / count * 100)

def build_dfa_from_dict(id, dict, labels):
    t = Trie(dict, labels)
    my_dfa = Dfa(id, t.states, t.arcs)
    # states are represented in a dfs fashion
    return my_dfa

def cosine_merging(dfa, states, states_mask, threshold):
    cos = torch.nn.CosineSimilarity(dim=-1)
    sim = cos(states[None, :, :], states[:, None, :])
    # sim1 = cos(states[None, states_mask, :], states[states_mask, None, :])
    # sim0 = cos(states[None, ~states_mask, :], states[~states_mask, None, :])

    total, pruned = 0, 0
    for i in range(states.shape[0]):
        for j in range(i):
            if (i == j):
                continue
            if (states_mask[i] != states_mask[j]):
                continue
            # if (states_mask[i] and sim1[i, j] > threshold) or (not states_mask[i] and sim0[i, j] > threshold):
            if (sim[i, j] > threshold):
                total += 1
                res = dfa.merge_states(i, j)
                pruned += 1 - res
    dfa.id = str(dfa.id) + 'min'
    # print("Found", total, "different pairs of states to be equivalent.", "Pruned", pruned)

    return dfa

def cross_validate(left, right, dfa, states, states_mask, val_sents, val_gold):

    max_acc = -1.
    # we run merging multiple times, and select the best
    for j in np.arange(left, right, .005):
        # cur_threshold = (left + right) / 2
        _dfa = deepcopy(dfa)
        merge_dfa = cosine_merging(_dfa, states, states_mask, j)
        cur_acc = score_all_prefixes(merge_dfa, val_sents, val_gold)
        if (cur_acc > max_acc):
            max_acc = cur_acc
            opt_threshold = j
            opt_dfa = deepcopy(merge_dfa)

    return opt_dfa, opt_threshold, max_acc

if __name__ == "__main__":
    args = parse_args()
    init_train_acc, init_dev_acc, train_acc, dev_acc = {}, {}, {}, {}
    n_merged_states, n_min_states = defaultdict(list), defaultdict(list)
    n_train = range(args.n_train_low, args.n_train_high)
    tokenizer = Tokenizer()
    tfilename = os.path.join("models", args.lang, "tokenizer.pkl")
    with open(tfilename, "rb") as fh:
        tokenizer = pickle.load(fh)
    lang = Language.from_string(args.lang)
    sampler = BalancedSampler(lang)
    dev_sampler = TestSampler(lang)
    if lang is None:
        raise NotImplementedError("Non implemented language.")

    for seed in range(args.seeds):
        random.seed(seed)
        init_train_acc[seed], init_dev_acc[seed], train_acc[seed], dev_acc[seed] = [], [], [], []
        for n in n_train:
            # n train and 1000 dev samples of length 10 and 50, respectively
            train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, n, args.train_length)
            val_tokens, val_labels, val_mask, val_sents = get_data(sampler, lang, tokenizer, 20, 2*args.train_length)
            dev_tokens, _dev_labels, dev_mask, dev_sents = get_data(dev_sampler, lang, tokenizer, 1000, 50)
            dev_labels = [_dev_labels[i][dev_mask[i]][-1] for i in range(len(_dev_labels))] # valid for TestSampler

            # Define the neural net and get predictions on the train/val/dev set
            trained_model = Tagger(tokenizer.n_tokens, 10, 100)
            filename = f"./models/{args.lang}/{args.epoch}.th"
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            trained_model.load_state_dict(torch.load(filename, map_location=device))
            with torch.no_grad():
                train_results = trained_model(train_tokens, train_labels, train_mask)
                val_results = trained_model(val_tokens, val_labels, val_mask)
                dev_results = trained_model(dev_tokens, _dev_labels, dev_mask)
            train_preds = train_results["predictions"]
            val_preds = val_results["predictions"]
            _dev_preds = dev_results["predictions"]
            dev_preds = [_dev_preds[i][dev_mask[i]][-1] for i in range(len(_dev_preds))] # valid for TestSampler

            # Can either use preds or labels here, depending on what we want to evaluate
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

            # Define the maximal dfa-trie
            redundant_dfa = build_dfa_from_dict(id=args.lang+str(args.sim_threshold), dict=train_sents, labels=train_gold) # build the trie based on train_gold
            assert(score_all_prefixes(redundant_dfa, train_sents, train_gold) == 100.) # 100% initial accuracy

            # Obtain representations
            representations = train_results["states"]
            idx = [redundant_dfa.return_states(sent) for sent in train_sents] # maps strings to states
            n_states = len(redundant_dfa.table.keys())
            states = torch.empty((n_states, 100))
            states_mask = torch.empty((n_states), dtype=torch.long)
            for i, _r in enumerate(representations):
                states[idx[i]] = _r[train_mask[i]]
                states_mask[idx[i]] = train_labels[i][train_mask[i]]

            init_dfa = deepcopy(redundant_dfa)
            if (args.find_threshold):
                # Merge states based on optimal similarity threshold
                # (measured by performance on hold out validation set (of length equal to 2 times the training length))
                merge_dfa, _, _ = cross_validate(.925, 1., redundant_dfa, states, states_mask, val_sents, val_gold)
                # print(opt_thr, max_dev_acc)
            else:
                # Merge states based on fixed similarity threshold
                merge_dfa = cosine_merging(redundant_dfa, states, states_mask, threshold=args.sim_threshold)

            if (args.min):
                merge_pdfa = to_pythomata_dfa(merge_dfa)
                min_pdfa = merge_pdfa.minimize().trim()
                min_pdfa = min_pdfa.minimize().trim()
                min_dfa = from_pythomata_dfa(min_pdfa)
                merge_dfa = min_dfa

            if (args.fst):
                init_dfa.make_graph()
                merge_dfa.make_graph()

            # Evaluate performance
            _acc = score_all_prefixes(merge_dfa, train_sents, train_gold)
            train_acc[seed].append(_acc)
            _acc = score_whole_words(merge_dfa, dev_sents, dev_gold) # valid for TestSampler
            dev_acc[seed].append(_acc)

            _acc = score_all_prefixes(init_dfa, train_sents, train_gold)
            init_train_acc[seed].append(_acc)
            _acc = score_whole_words(init_dfa, dev_sents, dev_gold) # valid for TestSampler
            init_dev_acc[seed].append(_acc)

            if not args.no_state_count:
                merge_pdfa = to_pythomata_dfa(merge_dfa)
                min_pdfa = merge_pdfa.minimize().trim()
                n_merged_states[seed].append(len(merge_pdfa.states))
                n_min_states[seed].append(len(min_pdfa.states))


    if (args.table):
        # Just generate percentages for the accuracy table
        dev_array = np.array(list(dev_acc.values()))
        mean_dev_acc = np.average(dev_array, axis=0)
        std_dev_acc = np.std(dev_array, axis=0)
        print(f"{args.train_length}--{args.sim_threshold}--{args.lang}--{mean_dev_acc}--{std_dev_acc}")
        exit()

    # Create plot for accuracy vs #data
    if (args.find_threshold):
        create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_train, args.lang, 'val-opt', args.epoch, args.eval)
    else:
        create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_train, args.lang, args.sim_threshold, args.epoch, args.eval)
