import torch
import argparse
from copy import deepcopy

from automaton import Dfa, equiv
from trie import Trie
from languages import *
from create_plot import create_plot
from models import Tagger
from utils import get_data, Tokenizer
from sampling import RandomSampler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    # parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--n_train_high", type=int, default=6)
    parser.add_argument("--n_train_low", type=int, default=5)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--fst", dest='fst', action='store_true')
    parser.add_argument("--no-fst", dest='fst', action='store_false')
    parser.set_defaults(fst=False)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.add_argument('--no-minimize', dest='min', action='store_false')
    parser.set_defaults(min=False)
    return parser.parse_args()

args = parse_args()

def score(dfa, dataset, labels):
    # Evaluate the performance of the extracted DFA on the dataset
    count, acc = 0, 0
    for i, word in enumerate(dataset):
        cur = ''
        for j, char in enumerate(word):
            acc += (dfa.accept(cur) == labels[i][j])
            cur += char
            count += 1
        acc += (dfa.accept(cur) == labels[i][j]) # complete word
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
n_train = range(args.n_train_low, args.n_train_high)
tokenizer = Tokenizer()
sampler = RandomSampler()

for seed in range(args.seeds):
    random.seed(seed)
    train_acc[seed], dev_acc[seed] = [], []
    for n in n_train:
        # n train and dev samples of length 50 and 100, respectively
        train_tokens, train_labels, train_mask, train_sents = get_data(sampler, lang, tokenizer, n, 50)
        _, dev_labels, _, dev_sents = get_data(sampler, lang, tokenizer, n, 100)

        # Define the maximal dfa-trie and the neural net
        redundant_dfa = build_dfa_from_dict(id=args.lang, dict=train_sents, labels=train_labels)
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
        if (args.min):
            min_dfa.minimize()
        # min_dfa.add_junk(alphabet=['a', 'b'])
        if (args.fst):
            init_dfa.make_graph()
            min_dfa.make_graph()

        # Evaluate performance
        _acc = score(min_dfa, train_sents, train_labels)
        train_acc[seed].append(_acc)
        _acc = score(min_dfa, dev_sents, dev_labels)
        dev_acc[seed].append(_acc)

# Create plot for accuracy vs #data
create_plot(train_acc, dev_acc, n_train, args.lang, args.sim_threshold)
