import unittest
import math

import torch
from peta.metrics.spearmanr import spearmanr


class TestSpearmanr(unittest.TestCase):
    def test_identical_tensors(self):
        # Test when pred and target are identical tensors
        pred = torch.tensor([1, 2, 3], dtype=torch.float)
        target = torch.tensor([1, 2, 3], dtype=torch.float)
        self.assertEqual(spearmanr(pred, target), 1.0)
        self.assertTrue(True)

    def test_opposite_tensors(self):
        # Test when pred and target are opposite tensors
        pred = torch.tensor([1, 2, 3], dtype=torch.float)
        target = torch.tensor([3, 2, 1], dtype=torch.float)
        self.assertEqual(spearmanr(pred, target), -1.0)

    def test_random_tensors(self):
        # Test when pred and target are random tensors
        pred = torch.randn(100)
        target = torch.randn(100)
        rho = spearmanr(pred, target)
        self.assertTrue(-1.0 <= rho <= 1.0)

    def test_empty_tensors(self):
        # Test when pred and target are empty tensors
        pred = torch.tensor([])
        target = torch.tensor([])
        self.assertTrue(math.isnan(spearmanr(pred, target)))


if __name__ == "__main__":
    unittest.main()
