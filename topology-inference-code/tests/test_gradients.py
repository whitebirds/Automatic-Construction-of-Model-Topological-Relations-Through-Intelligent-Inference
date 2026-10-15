"""Finite-difference checks for the manually differentiated neural modules."""
import unittest
import numpy as np
from topology_inference.core.character_hierarchical_network import check_gradients
from topology_inference.core.dedicated_bus_network import gradient_check
from topology_inference.core.learned_reference_study import MemoryNet


class NumericalGradientTests(unittest.TestCase):
    def test_character_and_hierarchical_network(self):
        self.assertLess(check_gradients()['max_error'], 2e-5)

    def test_bus_candidate_network(self):
        self.assertLess(gradient_check()['max_error'], 2e-5)

    def test_history_heads_and_null_gate(self):
        rng = np.random.default_rng(23)
        refs = [{'station': rng.normal(size=2), 'structure': rng.normal(size=(2, 3)),
                 'transfer': rng.normal(size=(2, 3, 2))} for _ in range(2)]
        task = {'base': np.array([.2, .3, .5]), 'refs': refs}
        model = MemoryNet([2, 3, 2], 19, 'structure')
        for memory_only in (False, True):
            _, gradients = model.loss_grad(task, 2, memory_only)
            for parameter, derivative in zip(model.params(), gradients):
                index = np.unravel_index(0, parameter.shape)
                old = parameter[index]
                parameter[index] = old + 1e-5
                plus = model.loss_grad(task, 2, memory_only)[0]
                parameter[index] = old - 1e-5
                minus = model.loss_grad(task, 2, memory_only)[0]
                parameter[index] = old
                self.assertAlmostEqual((plus - minus) / 2e-5, derivative[index], delta=2e-5)


if __name__ == '__main__':
    unittest.main()
