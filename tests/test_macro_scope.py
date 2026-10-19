import unittest
from rpv.evaluation.metrics import macro_f1


class MacroScopeTests(unittest.TestCase):
    def test_novel_denominator_and_base_false_negative(self):
        truth = ['new_a', 'new_a', 'new_b', 'new_b']
        predicted = ['new_a', 'base', 'new_b', 'new_b']
        self.assertAlmostEqual(macro_f1(truth, predicted, labels=['new_a', 'new_b']), 5 / 6)
        self.assertAlmostEqual(macro_f1(truth, predicted, labels=['base', 'new_a', 'new_b']), 5 / 9)
        with self.assertRaises(ValueError):
            macro_f1(truth, predicted, labels=['new_a', 'new_a'])
