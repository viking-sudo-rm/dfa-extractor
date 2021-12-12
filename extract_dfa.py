import torch
import argparse
from copy import deepcopy
import random
from collections import defaultdict

from automaton import Dfa, equiv
from pythomata_wrapper import to_pythomata_dfa
from trie import Trie
from languages import Language
from create_plot import create_plot
from models import Tagger
from utils import get_data, Tokenizer
from sampling import BalancedSampler, TestSampler
# from pythomata_wrapper import to_pythomata_nfa, to_pythomata_dfa, from_pythomata_dfa


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--n_train_low", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=6)
    parser.add_argument("--sim_threshold", type=float, default=.99)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fst", dest='fst', action='store_true')
    parser.add_argument("--no-fst", dest='fst', action='store_false')
    parser.set_defaults(fst=False)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.add_argument('--no-minimize', dest='min', action='store_false')
    parser.set_defaults(min=False)
    parser.add_argument('--epoch', type=str, default="best")
    parser.add_argument('--eval', type=str, default="preds")
    return parser.parse_args()

args = parse_args()

def score_whole_words(dfa, dataset, labels):
    acc = 0
    for word, y in zip(dataset, labels):
        acc += (dfa.accept(word) == y)
    return (acc / len(dataset) * 100)

def score_all_prefixes(dfa, dataset, labels):
    # Evaluate the performance of the extracted DFA on the dataset
    count, acc = 0, 0
    for i, word in enumerate(dataset):
        cur = ''
        for j, char in enumerate(word):
            acc += (dfa.accept(cur) == labels[i][j])
            cur += char
            count += 1
        if (cur != ''):
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

    total, pruned = 0, 0
    for i in range(states.shape[0]):
        for j in range(i):
            if (i == j):
                continue
            if (states_mask[i] != states_mask[j]):
                continue
            if (sim[i, j] > threshold):
                total += 1
                res = dfa.merge_states(i, j)
                pruned += 1 - res
    dfa.id = str(dfa.id) + 'min'
    # print("Found", total, "different pairs of states to be equivalent.", "Pruned", pruned)

    return dfa

# def minimize(auto: Dfa) -> Dfa:
#     nfa = to_pythomata_nfa(auto)
#     min_dfa = nfa.determinize().minimize().trim()
#     return from_pythomata_dfa(min_dfa)

init_train_acc, init_dev_acc, train_acc, dev_acc = {}, {}, {}, {}
n_merged_states, n_min_states = defaultdict(list), defaultdict(list)
n_train = range(args.n_train_low, args.n_train_high)
tokenizer = Tokenizer()
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
        train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, n, 10)
        # print(train_sents)
        dev_tokens, _dev_labels, dev_mask, dev_sents = get_data(dev_sampler, lang, tokenizer, 1000, 50)
        dev_labels = [_dev_labels[i][dev_mask[i]][-1] for i in range(len(_dev_labels))] # valid for TestSampler

        # Define the neural net and get predictions on the train/dev set
        trained_model = Tagger(tokenizer.n_tokens, 10, 100)
        filename = f"./models/{args.lang}/{args.epoch}.th"
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        trained_model.load_state_dict(torch.load(filename, map_location=device))
        with torch.no_grad():
            train_results = trained_model(train_tokens, train_labels, train_mask)
            dev_results = trained_model(dev_tokens, _dev_labels, dev_mask)
        train_preds = train_results["predictions"]
        _dev_preds = dev_results["predictions"]
        dev_preds = [_dev_preds[i][dev_mask[i]][-1] for i in range(len(_dev_preds))] # valid for TestSampler

        # Can either use preds or labels here, depending on what we want to evaluate
        # TODO: Maybe plot both on the same graph?
        if (args.eval == "preds"):
            train_gold = train_preds
            dev_gold = dev_preds
        elif (args.eval == "labels"):
            train_gold = train_labels
            dev_gold = dev_labels
        else:
            raise ValueError("Choose --eval between predictions `preds` and labels `labels`.")

        # Define the maximal dfa-trie
        redundant_dfa = build_dfa_from_dict(id=args.lang, dict=train_sents, labels=train_gold) # build the trie based on train_gold
        assert(score_all_prefixes(redundant_dfa, train_sents, train_gold) == 100.)

        # Obtain representations
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
        # The minimization is suuuper slow :(, probably because there is determinization first.
        # min_dfa = minimize(merge_dfa)

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

        merge_pdfa = to_pythomata_dfa(merge_dfa)
        min_pdfa = merge_pdfa.minimize().trim()
        n_merged_states[seed].append(len(merge_pdfa.states))
        n_min_states[seed].append(len(min_pdfa.states))
        # if n > 40:
        #     print("Yo got to stuff")
        #     pdfa = to_pythomata_dfa(merge_dfa)
        #     pdfa = pdfa.minimize().trim()
        #     print(pdfa.transition_function)
        #     break
        #     # graph.render("out.pdf")
        #     # break


# Create plot for accuracy vs #data
create_plot(init_train_acc, init_dev_acc, train_acc, dev_acc, n_train, args.lang, args.sim_threshold, args.epoch, args.eval)
