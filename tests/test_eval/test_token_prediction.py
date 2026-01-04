import unittest

import numpy as np
import torch
import torch.nn.functional as F

from flash_ansr.eval.metrics.token_prediction import (
    correct_token_predictions_at_k,
    reciprocal_rank, recall, precision,
    f1_score, accuracy, perplexity)


class TestCorrectTokenPredictions(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
        self.labels = torch.tensor([2, 1])
        self.k = 1

    def test_mean_reduction(self):
        result = correct_token_predictions_at_k(self.logits, self.labels, self.k, 'mean')
        self.assertEqual(result, torch.tensor(0.5))

    def test_sum_reduction(self):
        result = correct_token_predictions_at_k(self.logits, self.labels, self.k, 'sum')
        self.assertEqual(result, torch.tensor(1.0))

    def test_none_reduction(self):
        result = correct_token_predictions_at_k(self.logits, self.labels, self.k, 'none')
        self.assertTrue(torch.equal(result, torch.tensor([1., 0.])))

    def test_invalid_reduction(self):
        with self.assertRaises(ValueError):
            correct_token_predictions_at_k(self.logits, self.labels, self.k, 'invalid')

    def test_ignore_index_1(self):
        result = correct_token_predictions_at_k(self.logits, self.labels, self.k, ignore_index=1)
        self.assertEqual(result, torch.tensor(1.0))


class TestReciprocalRank(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]])
        self.labels = torch.tensor([2, 1])

    def test_mean_reduction(self):
        result = reciprocal_rank(self.logits, self.labels, 'mean')
        self.assertEqual(result, torch.tensor(0.75))

    def test_sum_reduction(self):
        result = reciprocal_rank(self.logits, self.labels, 'sum')
        self.assertEqual(result, torch.tensor(1.5))

    def test_none_reduction(self):
        result = reciprocal_rank(self.logits, self.labels, 'none')
        self.assertTrue(torch.equal(result, torch.tensor([1., .5])))

    def test_invalid_reduction(self):
        with self.assertRaises(ValueError):
            reciprocal_rank(self.logits, self.labels, 'invalid')

    def test_ignore_index_1(self):
        result = reciprocal_rank(self.logits, self.labels, ignore_index=1)
        self.assertEqual(result, torch.tensor(1.0))


class TestRecall(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3, 0, 3],
            [2, 3, 0]
        ])

        self.labels = torch.tensor([
            [0, 3, 1],  # Expected recall: 2/3 since two out of three predictions (id 0 and 3) appear in the labels
            [1, 1, 0]  # Expected recall: 0.5 since only one prediction (id 1) appears in the labels
        ])

    def test_no_reduction(self):
        result = recall(self.logits, self.labels, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([2 / 3, 0.5])))

    def test_ignore_index(self):
        result = recall(self.logits, self.labels, ignore_index=0, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([0.5, 0])))

    def test_mean_reduction_ignore_index(self):
        result = recall(self.logits, self.labels, ignore_index=0, reduction='mean')
        self.assertEqual(result, torch.tensor(0.25))


class TestPrecision(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3, 0, 3],
            [2, 3, 0]
        ])

        self.labels = torch.tensor([
            [0, 3, 1],  # Expected precision: 1 since all the predictions appear in the labels
            [1, 1, 0]  # Expected precision: 1/3 since only one prediction (id 0) appears in the labels
        ])

    def test_no_reduction(self):
        result = precision(self.logits, self.labels, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([1., 1 / 3])))

    def test_ignore_index(self):
        result = precision(self.logits, self.labels, ignore_index=0, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([1., 0])))

    def test_mean_reduction_ignore_index(self):
        result = precision(self.logits, self.labels, ignore_index=0, reduction='mean')
        self.assertEqual(result, torch.tensor(0.5))


class TestF1Score(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3, 0, 3],
            [2, 3, 0]
        ])

        self.labels = torch.tensor([
            [0, 3, 1],  # Expected F1 score: = 2 * (1 * 2/3) / (1 + 2/3) = 4/5
            [1, 1, 0]  # Expected F1 score: = 2 * (1/3 * 0.5) / (1/3 + 0.5) = 2/5
        ])

    def test_no_reduction(self):
        result = f1_score(self.logits, self.labels, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([4 / 5, 2 / 5])))


class TestMetricInputFlexibility(unittest.TestCase):
    def test_python_lists_batch_inputs(self):
        preds = [[1, 2, 3], [3, 4, 5]]
        labels = [[3, 2, 1], [4, 5, 6]]

        expected_precision = torch.tensor([1.0, 2 / 3], dtype=torch.float32)
        expected_recall = torch.tensor([1.0, 2 / 3], dtype=torch.float32)

        precision_values = precision(preds, labels, reduction='none')
        recall_values = recall(preds, labels, reduction='none')

        self.assertTrue(torch.allclose(precision_values, expected_precision))
        self.assertTrue(torch.allclose(recall_values, expected_recall))

    def test_single_sequence_numpy_arrays(self):
        preds = np.array([1, 2, 2, 0])
        labels = np.array([2, 3, 0])

        precision_value = precision(preds, labels, ignore_index=0, reduction='mean')
        recall_value = recall(preds, labels, ignore_index=0, reduction='mean')
        f1_value = f1_score(preds, labels, ignore_index=0, reduction='mean')

        self.assertTrue(torch.allclose(precision_value, torch.tensor(0.5)))
        self.assertTrue(torch.allclose(recall_value, torch.tensor(0.5)))
        self.assertTrue(torch.allclose(f1_value, torch.tensor(0.5)))

    def test_string_tokens_with_ignore(self):
        preds = [["cat", "dog", "PAD"], ["apple", "banana", "banana"]]
        labels = [["dog", "mouse", "PAD"], ["banana", "pear"]]

        expected = torch.tensor([0.5, 0.5], dtype=torch.float32)

        precision_values = precision(preds, labels, ignore_index="PAD", reduction='none')
        recall_values = recall(preds, labels, ignore_index="PAD", reduction='none')
        f1_values = f1_score(preds, labels, ignore_index="PAD", reduction='none')

        self.assertTrue(torch.allclose(precision_values, expected))
        self.assertTrue(torch.allclose(recall_values, expected))
        self.assertTrue(torch.allclose(f1_values, expected))


class TestAccuracy(unittest.TestCase):
    def setUp(self):
        self.logits = torch.tensor([
            [3, 3, 1],
            [2, 3, 0]
        ])

        self.labels = torch.tensor([
            [0, 3, 1],  # Accuracy: 0 since the sequences do not match
            [2, 3, 0]  # Accuracy: 1 since the sequences match
        ])

    def test_no_reduction(self):
        result = accuracy(self.logits, self.labels, reduction='none')
        self.assertTrue(torch.equal(result, torch.tensor([0., 1.])))

    def test_ignore_index(self):
        result = accuracy(self.logits, self.labels, ignore_index=0, reduction='none')
        print(result)
        self.assertTrue(torch.equal(result, torch.tensor([0., 1.])))

    def test_mean_reduction_ignore_index(self):
        result = accuracy(self.logits, self.labels, ignore_index=0, reduction='mean')
        self.assertEqual(result, torch.tensor(0.5))


class TestPerplexity(unittest.TestCase):
    def test_perplexity_mean_reduction(self):
        # Define sample logits and labels
        logits = torch.tensor([[[2.0, 0.5, 0.3], [0.1, 1.2, 3.1]],
                               [[2.5, 1.5, 0.2], [1.1, 0.2, 0.3]]])
        labels = torch.tensor([[0, 2], [1, 2]])

        # Compute expected cross-entropy loss
        flattened_logits = logits.view(-1, logits.size(-1))
        flattened_labels = labels.view(-1)
        cross_entropy_loss = F.cross_entropy(flattened_logits, flattened_labels, reduction='none')

        # Compute expected perplexity
        expected_perplexity = torch.exp(cross_entropy_loss).mean().item()

        # Compute actual perplexity using the function
        actual_perplexity = perplexity(logits, labels, reduction='mean').item()

        # Assert the values are close
        self.assertAlmostEqual(expected_perplexity, actual_perplexity, places=5)
