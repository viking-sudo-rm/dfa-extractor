from typing import Iterable
import subprocess


class Dfa:
    """
    A basic class that represents a deterministic finite automaton.
    """

    def __init__(self, id = 0, states = [], arcs = [], init_state=0, nfa=False):
        """
        Expects an identifier (used for figure creation), a list of states in the
        form of tuples (n, is_final_state), and a list of tuples (state1, symbols, state2)
        for the transitions.
        """

        self.table = {}
        self.final = {}
        self.init_state = init_state
        self.id = id
        self.nfa = nfa
        for v in states:
            self.table[v[0]] = []
            self.final[v[0]] = v[1]

        for e in arcs:
            self.table[e[0]].append([e[1], e[2]]) # transition symbol and then output state

    def add_junk(self, alphabet):
        first_time = True
        # force copy
        nodes = list(self.table.keys())
        arcs = list(self.table.values())
        for state, transitions in zip(nodes, arcs):
            for char in alphabet:
                if (char not in [e[0] for e in transitions]):
                    if (first_time):
                        first_time = False
                        n = max([k for k in self.table.keys()])
                        self.final[n+1] = False
                        self.table[n+1] = [[char, n+1] for char in alphabet]
                    self.table[state].append([char, n+1])

    def accept(self, string: Iterable[str]):
        """
        Determines whether the automaton accepts or not a string.
        """

        # TODO: merge the two cases
        if (not self.nfa):
            cur_state = self.init_state
            for s in string:
                found = False
                for arcs in self.table[cur_state]:
                    if (arcs[0] == s):
                        cur_state = arcs[1]
                        found = True
                if (not found):
                    return False
            return self.final[cur_state]
        else:
            prev_states = [self.init_state]
            for s in string:
                found = False
                cur_states = []
                for prev in prev_states:
                    for arcs in self.table[prev]:
                        if (arcs[0] == s and not (arcs[1] in cur_states)):
                            cur_states.append(arcs[1])
                if (len(cur_states) == 0):
                    return False
                prev_states = cur_states
            return any([self.final[_cur] for _cur in prev_states])

    def return_states(self, string):
        states = [0] * (len(string) + 1)
        cur_state = self.init_state
        for i, s in enumerate(string):
            for arcs in self.table[cur_state]:
                if (arcs[0] == s):
                    states[i] = cur_state
                    cur_state = arcs[1]


        if (cur_state != self.init_state):
            states[i+1] = cur_state
        return states

    def merge_states(self, state1, state2, symmetric=False):
        """
        Merges state1 and state2.
        """

        if (state1 not in self.table.keys() or state2 not in self.table.keys()):
            # already pruned state
            return 1

        if (symmetric):
            for arcs in self.table[state2]:
                if (any([arcs[0] == x[0] for x in self.table[state1]])):
                    return 1

        # if (self.final[state1] != self.final[state2]):
        #     # print("Trivially wrong. One state is final while the other is not.")
        #     return 1

        # find ingoing of the second state
        for w in self.table.values():
            for arcs in w:
                if (arcs[1] == state2):
                    arcs[1] = state1 # point to the first state

        # find outgoing of the second state
        for arcs in self.table[state2]:
            if (not self.nfa and any([arcs[0] == x[0] for x in self.table[state1]])): # if there is a conflict, continue (we only keep state1's transition)
                continue
            # if (arcs[1] == state1): # we don't want to create self loops or do we?
            #     continue
            self.table[state1].append([arcs[0], arcs[1]])

        # self.final[state1] = self.final[state2] or self.final[state1] # arbitrary (does it makes sense?)
        if (state2 == self.init_state):
            self.init_state = state1
        del self.table[state2]
        del self.final[state2]
        return 0

    def make_graph(self):
        """
        Creates a pdf file for the automaton.
        """

        text_filename = str(self.id) + '.txt.fst'
        with open(text_filename, 'w') as f:
            for v, w in self.table.items():
                for arc in w:
                    f.write(str(v) + ' ' + str(arc[1]) + ' ' + str(arc[0]) + '\n')
                    # f.write(str(v) + ' ' + str(arc[1]) + ' ' + str(arc[0]) + ' ' + '<eps>' + '\n')
            for v, is_final in self.final.items():
                if (is_final):
                    f.write(str(v) + '\n')

        # requires openfst
        binary_filename = str(self.id) + '.fst'
        subprocess.run(["fstcompile", "--acceptor", "--isymbols=isyms.txt",
                        "--osymbols=osyms.txt", "--keep_isymbols", "--keep_osymbols",
                        text_filename, binary_filename])
        subprocess.run(["fstdraw", binary_filename, str(self.id) + '.dot'])
        out = open("./images/" + str(self.id) + ".pdf", 'w')
        subprocess.run(["dot", "-Tpdf", str(self.id) + '.dot'], stdout=out)
        out.close()

    def minimize(self):
        """
        Minimize current dfa (the .fst file, not the dfa that is represented from this class)
        (using openfst's fstminimize)
        """
        # print(str)
        res = subprocess.run(["fstminimize", str(self.id) + '.fst', str(self.id) + '.fst'], capture_output=True)

def equiv(dfa1, dfa2):
    """
    Determines whether two DFAs are equivalent
    (using openfst's fstequivalent).
    """

    name1, name2 = str(dfa1.id) + '.fst', str(dfa2.id) + '.fst'
    res = subprocess.run(["fstequivalent", name1, name2], capture_output=True)
    # print(res)
    return res.returncode
