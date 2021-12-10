from automaton import Dfa, equiv
from trie import Trie
import torch
from languages import *
import argparse
from models import Tagger, LanguageModel
from utils import sequence_cross_entropy_with_logits, get_data, Tokenizer
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def plot_creation(train_acc, dev_acc, lengths, lang, threshold):

    train_array = np.array(list(train_acc.values()))
    mean_train_acc = np.average(train_array, axis=0)
    std_train_acc = np.std(train_array, axis=0)
    dev_array = np.array(list(dev_acc.values()))
    mean_dev_acc = np.average(dev_array, axis=0)
    std_dev_acc = np.std(dev_array, axis=0)

    fig, ax = plt.subplots()
    ax.plot(lengths, mean_train_acc, linestyle='-', marker='.', label='train acc')
    plt.plot(lengths, mean_dev_acc, linestyle='-', marker='.', label='dev acc')
    ax.fill_between(lengths, mean_train_acc - std_train_acc, mean_train_acc + std_train_acc, alpha = 0.3)
    plt.fill_between(lengths, mean_dev_acc - std_dev_acc, mean_dev_acc + std_dev_acc, alpha = 0.3)
    ax.set_xlabel("#data")
    ax.set_ylabel("Accuracy")
    plt.title(args.lang + ', threshold =' + str(args.sim_threshold))
    ax.legend()
    plotname = f"./images/{args.lang + str(args.sim_threshold)}.pdf"
    # plt.savefig(plotname)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    # parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--upper_length", type=int, default=5)
    parser.add_argument("--lower_length", type=int, default=5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.add_argument('--no-minimize', dest='min', action='store_false')
    parser.set_defaults(min=False)
    return parser.parse_args()

args = parse_args()

def build_fsa_from_dict(id, dict):
    t = Trie(dict)
    my_dfa = Dfa(id, t.states, t.arcs)
    # states are represented in a dfs fashion
    return my_dfa

def cosine_merging(dfa, states, states_mask, threshold):
    cos = torch.nn.CosineSimilarity(dim=-1)
    total, pruned = 0, 0
    # print(states.shape)
    # print((states @ states.transpose(0, 1)).shape)
    # x = torch.randn(32, 100, 25)
    sim = cos(states[None, :, :], states[:, None, :])
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
    print("Found", total, "different pairs of states to be equivalent.", "Pruned", pruned)

    return dfa

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

train_acc, dev_acc = {}, {}
lengths = range(args.lower_length, args.upper_length)
tokenizer = Tokenizer()


for seed in range(1):
    random.seed(seed)
    train_acc[seed], dev_acc[seed] = [], []
    for n in lengths:
        train_tokens, train_labels, train_mask, train_sents = get_data(lang, tokenizer, 0, n)
        _, _, _, dev_sents = get_data(lang, tokenizer, n+1, n+101)

        redundant_dfa = build_fsa_from_dict(id=args.lang, dict=train_sents)
        redundant_dfa.make_graph()

        trained_model = Tagger(tokenizer.n_tokens, 10, 100)
        filename = f"./models/best{args.lang}.th"
        trained_model.load_state_dict(torch.load(filename))

        # Obtain representations
        representations = trained_model(train_tokens, train_labels, train_mask)["states"]
        idx = [redundant_dfa.return_states(sent) for sent in train_sents]
        n_states = len(redundant_dfa.table.keys())
        states = torch.empty((n_states, 100))
        states_mask = torch.empty((n_states), dtype=torch.long)
        for i, _r in enumerate(representations):
            states[idx[i]] = _r[train_mask[i]]
            states_mask[idx[i]] = train_labels[i][train_mask[i]]

        # Merge states
        init_dfa = deepcopy(redundant_dfa)
        min_dfa = cosine_merging(redundant_dfa, states, states_mask, threshold=args.sim_threshold)
        init_dfa.make_graph()
        if (args.min):
            min_dfa.minimize()
        # min_dfa.add_junk(alphabet=['a', 'b'])
        min_dfa.make_graph()

        acc = 0
        for word in train_sents:
            acc += min_dfa.accept(word)

        train_acc[seed].append(acc / len(train_sents) * 100)

        acc = 0
        # if (args.lang == "Tom6"):
        #     gen = lang.generate(n+6, n+106)
        # else:
        #     gen = lang.generate(n+2, n+102)
        for word in dev_sents:
            acc += min_dfa.accept(word)
        dev_acc[seed].append(acc)

plot_creation(train_acc, dev_acc, lengths, args.lang, args.sim_threshold)
