from unittest import TestCase

from automaton import Dfa
from pythomata_wrapper import to_pythomata


class TestPythomataWrapper(TestCase):
    
    def test_extract(self):
        states = [(0, True), (1, False), (2, True), (3, False)]
        arcs = [(0, "a", 1), (1, "b", 2), (2, "a", 3), (3, "b", 0)]
        dfa = Dfa(states=states, arcs=arcs)
        pdfa = to_pythomata(dfa)
        self.assertEqual(pdfa.states, {0, 1, 2, 3})
        self.assertEqual(pdfa.initial_state, 0)
        self.assertEqual(pdfa.accepting_states, {0, 2})
        
        mdfa = pdfa.minimize().trim()
        self.assertEqual(mdfa.states, {0, 1})
        self.assertEqual(mdfa.initial_state, 0)
        self.assertEqual(mdfa.accepting_states, {0})
        self.assertEqual(mdfa.transition_function, {0: {'a': 1}, 1: {'b': 0}})
