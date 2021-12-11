from unittest import TestCase

from pythomata.impl.simple import SimpleDFA
from pythomata.impl.simple import SimpleNFA

from automaton import Dfa
from pythomata_wrapper import to_pythomata_nfa, from_pythomata_dfa


class TestPythomataWrapper(TestCase):
    
    def test_to_pythomata_nfa(self):
        states = [(0, True), (1, False), (2, True), (3, False)]
        arcs = [(0, "a", 1), (1, "b", 2), (2, "a", 3), (3, "b", 0)]
        dfa = Dfa(states=states, arcs=arcs)
        nfa = to_pythomata_nfa(dfa)
        self.assertEqual(nfa.states, {0, 1, 2, 3})
        self.assertEqual(nfa.initial_state, 0)
        self.assertEqual(nfa.accepting_states, {0, 2})
        self.assertEqual(nfa.transition_function, {0: {'a': {1}}, 1: {'b': {2}}, 2: {'a': {3}}, 3: {'b': {0}}})
        
        # After determinization, the states get weirdly renumbered.
        mdfa = nfa.determinize().minimize().trim()
        self.assertEqual(mdfa.states, {1, 3})
        self.assertEqual(mdfa.initial_state, 3)
        self.assertEqual(mdfa.accepting_states, {3})
        self.assertEqual(mdfa.transition_function, {1: {'b': 3}, 3: {'a': 1}})

    def test_from_pythomata_dfa(self):
        pdfa = SimpleDFA({0, 1}, {"a", "b"}, 0, {0}, {0: {'a': 1}, 1: {'b': 0}})
        dfa = from_pythomata_dfa(pdfa)
        self.assertEqual(dfa.init_state, 0)
        self.assertEqual(dfa.table, {0: [['a', 1]], 1: [['b', 0]]})

    def test_from_pythomata_nfa(self):
        pdfa = SimpleNFA({0, 1}, {"a", "b"}, 0, {0}, {0: {'a': {0, 1}}, 1: {'b': {0}}})
        dfa = from_pythomata_dfa(pdfa)
        self.assertEqual(dfa.init_state, 0)
        self.assertEqual(dfa.table, {0: [['a', 0], ['a', 1]], 1: [['b', 0]]})
