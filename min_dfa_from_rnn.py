from automaton import Dfa, equiv
from trie import Trie
import torch
from torch.nn.utils.rnn import pad_sequence
from languages import *
import argparse
from utils import sequence_cross_entropy_with_logits, LanguageModel, Tokenizer
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    # parser.add_argument("--length", type=int, default=2)
    parser.add_argument("--upper_length", type=int, default=10)
    parser.add_argument("--lower_length", type=int, default=2)
    parser.add_argument("--sim_threshold", type=float, default=0.5)
    parser.add_argument('--minimize', dest='min', action='store_true')
    parser.add_argument('--no-minimize', dest='min', action='store_false')
    parser.set_defaults(min=False)
    return parser.parse_args()

args = parse_args()

def build_fsa_from_dict(id, dict):
    t = Trie(dict)
    # print(t.strings)
    my_dfa = Dfa(id, t.states, t.arcs)
    # states are represented in a dfa fashion
    return my_dfa

def cosine_merging(dfa, states, threshold):
    cos = torch.nn.CosineSimilarity(dim=0)
    total, pruned = 0, 0
    for i in range(states.shape[0]):
        for j in range(i):
            if (i == j):
                continue
            if (cos(states[i], states[j]) > threshold):
                total += 1
                # print("The above gets checked")
                res = dfa.merge_states(i, j)
                pruned += 1 - res
                # print("The above got pruned?", res)
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
lengths = range(args.lower_length, args.upper_length)
for seed in range(1):
    random.seed(seed)
    train_acc[seed], dev_acc[seed] = [], []
    for n in lengths:
        if (args.lang == "Tom6"):
            train = list(lang.generate(4, n+5))
        else:
            train = list(lang.generate(0, n+1))
        train.sort(key=len, reverse=True)
        # print(train)
        max_length = len(train[0])
        arb = train[0][-1]

        redundant_dfa = build_fsa_from_dict(id=args.lang, dict=train)
        redundant_dfa.make_graph()

        tokenizer = Tokenizer()
        input = pad_sequence([torch.tensor(tokenizer.tokenize(sent + arb)) for sent in train], batch_first=True) # arbitrary symbol in the end so that we get final state

        # print(max_length, len(input[0]))
        trained_model = LanguageModel(tokenizer.n_tokens, 10, 100)
        filename = f"./models/best{args.lang}.th"
        trained_model.load_state_dict(torch.load(filename))

        n_states = len(redundant_dfa.table.keys())
        states = torch.empty((n_states, 100))
        for i, seq in enumerate(input):
            seq = seq.reshape(1, -1)
            mask = (seq != 0)
            representations = trained_model(seq, mask)['states']
            # print(seq.shape, len(train[i]))
            # print(representations.shape)
            idx = redundant_dfa.return_states(train[i])
            # print(mask)
            # print(len(idx))

            if (len(train[i]) == max_length):
                states[idx] = representations.squeeze()
            else:
                states[idx] = representations.squeeze()[mask.flatten()[:-1]][:-1] # because of arbitrary symbol

        init_dfa = deepcopy(redundant_dfa)
        min_dfa = cosine_merging(redundant_dfa, states, threshold=args.sim_threshold)
        init_dfa.make_graph()
        if (args.min):
            min_dfa.minimize()
        min_dfa.make_graph()

        acc = 0
        for word in train:
            acc += min_dfa.accept(word)
        # print(acc, len(train))
        train_acc[seed].append(acc / len(train) * 100)

        acc = 0
        if (args.lang == "Tom6"):
            gen = lang.generate(n+6, n+106)
        else:
            gen = lang.generate(n+2, n+102)
        for word in gen:
            acc += min_dfa.accept(word)
        dev_acc[seed].append(acc)

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
ax.set_xlabel("# data")
ax.set_ylabel("Accuracy")
plt.title(args.lang + ', threshold =' + str(args.sim_threshold))
ax.legend()
plotname = f"./images/{args.lang + str(args.sim_threshold)}_withoutselflo.pdf"
plt.savefig(plotname)
plt.show()
