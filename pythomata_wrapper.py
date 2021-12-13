from typing import Union

from pythomata import SimpleDFA
from pythomata.impl.simple import SimpleNFA

from automaton import Dfa


def to_pythomata_nfa(dfa: Dfa) -> SimpleNFA:
    """Convert our kind of DFA to a Pythomata DFA."""
    states = set(dfa.table.keys())
    accepting_states = {s for s, v in dfa.final.items() if v}
    alphabet = set()
    transition_function = {}
    for state, pairs in dfa.table.items():
        transition_function[state] = {}
        for token, new_state in pairs:
            alphabet.add(token)
            if token not in transition_function[state]:
                transition_function[state][token] = set()
            transition_function[state][token].add(new_state)
    return SimpleNFA(states, alphabet, dfa.init_state, accepting_states, transition_function)


def to_pythomata_dfa(dfa: Dfa) -> SimpleDFA:
    """Given a deterministic automaton, get deterministic one."""
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


def from_pythomata_dfa(auto: Union[SimpleDFA, SimpleNFA]) -> Dfa:
    """Converts a Pythomata DFA to our kind of DFA."""
    states = [(s, s in auto.accepting_states) for s in auto.states]
    arcs = []
    for state, mapping in auto.transition_function.items():
        if isinstance(auto, SimpleDFA):
            for token, new_state in mapping.items():
                arcs.append((state, token, new_state))
        else:
            for token, new_states in mapping.items():
                for new_state in new_states:
                    arcs.append((state, token, new_state))
    return Dfa('pmin', states, arcs, auto.initial_state)
