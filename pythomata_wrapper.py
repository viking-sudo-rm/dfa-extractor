from pythomata import SimpleDFA

from automaton import Dfa


def to_pythomata(dfa: Dfa) -> SimpleDFA:
    """Convert our kind of DFA to a Pythomata DFA."""
    states = set(dfa.table.keys())
    accepting_states = {s for s, v in dfa.final.items() if v}
    alphabet = set()
    transition_function = {}
    for state, pairs in dfa.table.items():
        transition_function[state] = {}
        for token, new_state in pairs:
            alphabet.add(token)
            transition_function[state][token] = new_state
    return SimpleDFA(states, alphabet, dfa.init_state, accepting_states, transition_function)
