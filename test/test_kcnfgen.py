import os
import sys
import unittest

from pysat.formula import CNF

from mlbf import kcnfgen

class TestDataset(unittest.TestCase):
    def test_phase_transition(self):
        self.assertEqual(91, kcnfgen.phase_transition_clauses(20))  # hardcoded
        self.assertEqual(218, kcnfgen.phase_transition_clauses(50))  # hardcoded
        self.assertEqual(322, kcnfgen.phase_transition_clauses(75))  # calculated
        self.assertEqual(431, kcnfgen.phase_transition_clauses(100))  # hardcoded
        self.assertEqual(534, kcnfgen.phase_transition_clauses(125))  # calculated
        self.assertEqual(645, kcnfgen.phase_transition_clauses(150))  # hardcoded
        self.assertEqual(747, kcnfgen.phase_transition_clauses(175))  # calculated
        self.assertEqual(854, kcnfgen.phase_transition_clauses(200))  # hardcoded
        self.assertEqual(959, kcnfgen.phase_transition_clauses(225))  # calculated


if __name__ == '__main__':
    unittest.main()
