from automaton import DFA, equiv
from trie import Trie
import torch
from torch.nn.utils.rnn import pad_sequence
from languages import *
import argparse
from utils import sequence_cross_entropy_with_logits, LanguageModel, Tokenizer
from copy import deepcopy

def build_fsa_from_dict(id, dict):
    t = Trie(dict)
    my_dfa = DFA(id, t.states, t.arcs)
    return my_dfa

corpus = ['aab', 'ab']
dfa = build_fsa_from_dict(id='tbd', dict=corpus)
dfa.make_graph()

# dfa1 = DFA(id = 42, states = [(0, False), (1, True)], arcs = [(0, 'a', 1)])
# # print(dfa1.final)
# print(dfa1.score('a'))
# print(dfa1.score('b'))
# print(dfa1.score('abbbbbb'))
#
# dfa1.merge_states(0, 1)

# dfa1.make_graph()

# dfa2 = DFA(id = 10, states = [(0, False), (1, False), (2, True), (3, False), (4, True)],
#                 arcs = [(0, 'a', 1), (1, 'b', 2), (2, 'a', 3), (3, 'b', 4), (4, 'a', 3)])
# print(dfa2.score('a'))
# print(dfa2.score('ab'))
# print(dfa2.score('abab'))
# print(dfa2.score('ab1'))
# dfa2.merge_states(1, 3)
# dfa2.merge_states(2, 4)
# print(dfa2.score('a'))
# print(dfa2.score('ab'))
# print(dfa2.score('abab'))
# print(dfa2.score('ab1'))
#
# # dfa2.make_graph()
#
# assert (not equiv(dfa1, dfa2))
