from automaton import DFA, equiv
from trie import Trie
import torch
from torch.nn.utils.rnn import pad_sequence
from abstar import AbstarGenerator
from utils import sequence_cross_entropy_with_logits, LanguageModel, Tokenizer

def build_fsa_from_dict(id, dict):
    t = Trie(dict)
    my_dfa = DFA(id, t.states, t.arcs)
    return my_dfa

def cosine_merging(dfa, states, threshold):
    cos = torch.nn.CosineSimilarity(dim=0)
    total, pruned = 0, 0
    for i in range(states.shape[1] - 1):
        for j in range(i+1):
            if (i == j):
                continue
            # print("States", i, j, "have similarity:", round(cos(states[0][i], states[0][j]).item(), 2))
            if (cos(states[0][i], states[0][j]) > threshold):
                total += 1
                res = dfa.merge_states(i+1, j+1)
                pruned += 1 - res
    dfa.id = str(dfa.id) + 'min'
    print("Found", total, "different pairs of states to be equivalent.", "Pruned", pruned)

    return dfa

n = 8
lang = AbstarGenerator()
redundant_dfa = build_fsa_from_dict(id=42, dict=lang.generate(1, n))
redundant_dfa.make_graph()

trained_model = LanguageModel(4, 10, 100)
trained_model.load_state_dict(torch.load("./models/best.th"))
tokenizer = Tokenizer()
input = pad_sequence([torch.tensor(tokenizer.tokenize('ab' * n + 'a'))], batch_first=True)
mask = (input != 0)

states = trained_model(input, mask)['states']

min_dfa = cosine_merging(redundant_dfa, states, threshold=0)
min_dfa.make_graph()

if (equiv(redundant_dfa, min_dfa) == 0):
    print("The two automata are equivalent.")
else:
    print("The two automata are not equivalent.")
