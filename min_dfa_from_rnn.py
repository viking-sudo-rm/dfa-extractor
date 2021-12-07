from automaton import DFA, equiv
from trie import Trie
import torch
from torch.nn.utils.rnn import pad_sequence
from languages import *
import argparse
from utils import sequence_cross_entropy_with_logits, LanguageModel, Tokenizer
from copy import deepcopy

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", type=str, default="Tom2")
    parser.add_argument("--length", type=int, default=4)
    parser.add_argument("--sim_threshold", type=float, default=0.)
    return parser.parse_args()

args = parse_args()

def build_fsa_from_dict(id, dict):
    t = Trie(dict)
    my_dfa = DFA(id, t.states, t.arcs)
    return my_dfa

def cosine_merging(dfa, states, threshold):
    cos = torch.nn.CosineSimilarity(dim=0)
    total, pruned = 0, 0
    for i in range(states.shape[1]):
        for j in range(i+1):
            print(i, j)
            if (i == j):
                continue
            if (i == 8 and j == 0):
                print(cos(states[0][i], states[0][j]))
            if (cos(states[0][i], states[0][j]) > threshold):
                total += 1
                print("The above gets checked")
                res = dfa.merge_states(i, j)
                pruned += 1 - res
                print("The above got pruned?", res)
    dfa.id = str(dfa.id) + 'min'
    print("Found", total, "different pairs of states to be equivalent.", "Pruned", pruned)

    return dfa

n = args.length
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

redundant_dfa = build_fsa_from_dict(id=args.lang, dict=lang.generate(1, n+1))

tokenizer = Tokenizer()
input = pad_sequence([torch.tensor(tokenizer.tokenize(seq * n))], batch_first=True) # it doesn't work here
mask = (input != 0)

trained_model = LanguageModel(tokenizer.n_tokens, 10, 100)
filename = f"./models/best{args.lang}.th"
trained_model.load_state_dict(torch.load(filename))

states = trained_model(input, mask)['states']

init_dfa = deepcopy(redundant_dfa)
min_dfa = cosine_merging(redundant_dfa, states, threshold=args.sim_threshold)
init_dfa.make_graph()
min_dfa.make_graph()

if (equiv(init_dfa, min_dfa) == 0):
    print("The two automata are equivalent.")
else:
    print("The two automata are not equivalent.")
