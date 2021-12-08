import subprocess

class Dfa:
    """
    A basic class that represents a deterministic finite automaton.
    """

    def __init__(self, id = 0, states = [], arcs = []):
        """
        Expects an identifier (used for figure creation), a list of states in the
        form of tuples (n, is_final_state), and a list of tuples (state1, symbols, state2)
        for the transitions.
        """

        self.table = {}
        self.final = {}
        self.init_state = 0
        self.id = id
        for v in states:
            self.table[v[0]] = []
            self.final[v[0]] = v[1]

        for e in arcs:
            self.table[e[0]].append([e[1], e[2]]) # transition symbol and then output state

    def score(self, string):
        """
        Determines whether the automaton accepts or not a string.
        """

        cur_state = self.init_state
        for s in string:
            found = False
            for arcs in self.table[cur_state]:
                if (arcs[0] == s):
                    cur_state = arcs[1]
                    found = True
            if (not found):
                return "FSA rejects"
        if (self.final[cur_state]):
            return "FSA accepts"
        else:
            return "FSA rejects"

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


    def merge_states(self, state1, state2):
        """
        Merges state1 and state2.
        """

        if (state1 not in self.table.keys() or state2 not in self.table.keys()):
            # already pruned state
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
            if (any([arcs[0] == x[0] for x in self.table[state1]])): # if there is a conflict, continue (we only keep state1's transition)
                continue
            self.table[state1].append([arcs[0], arcs[1]])

        # self.final[state1] = self.final[state2] or self.final[state1] # arbitrary (does it makes sense?)
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