import pandas as pd
from pysat.formula import CNF
import unittest

from mlsat import dataset


class TestDataset(unittest.TestCase):
    def test_prepare_dataset_simple(self):
        """
        Tests the preparation of a dataset from simple instances
        :return:
        """

        # suppose the formula: (-1v2) ^ (-1v3)
        positives = [
            [-1, 2, -3],
            [-1, -2, 3]
        ]
        negatives = [
            [1, -2, 3],
            [1, -2, -3]
        ]
        generator = dataset.DatasetGenerator()
        data_x, data_y = generator.prepare_dataset(positives, negatives)
        self.assertEqual(4, len(data_x))
        self.assertEqual(4, len(data_y))

        # creates a new dataframe appending data_x and y column-wise
        df = data_x.copy()
        df['y'] = data_y
        # print(df)

        # checking the presence of the instances
        self.assertIn([0, 1, 0, 1], df.values)  # -1,2,3 positive
        self.assertIn([0, 0, 1, 1], df.values)  # -1,-2,3 positive
        self.assertIn([1, 0, 1, 0], df.values)  # 1,-2,3 negative
        self.assertIn([1, 0, 0, 0], df.values)  # 1,-2,-3 negative

    def test_unigen_negative_simple(self):
        f = CNF(from_clauses=[[-1, 2], [3]])
        expected_negatives = {  # set of tuples, contains all unsat assignments for the formula
            (1, -2, 3),
            (-1, -2, -3),
            (-1, 2, -3),
            (1, -2, -3),
            (1, 2, -3)
        }

        sampler = dataset.UnigenDatasetGenerator()
        negatives = sampler.generate_negative_samples(f, 5)  # 5 is the max number of negatives
        self.assertEqual(5, len(negatives))

        # checks if all negatives are unique: transform into set and see if the length did not reduce
        neg_set = set(tuple(neg) for neg in negatives)
        self.assertEqual(5, len(neg_set))

        # checks if the returned set of negatives is correct
        self.assertEqual(expected_negatives, neg_set)

    def test_unigen_negative_max_attempts(self):
        f = CNF(from_clauses=[[-1, 2], [3]])
        expected_negatives = {  # set of tuples, contains all unsat assignments for the formula
            (1, -2, 3),
            (-1, -2, -3),
            (-1, 2, -3),
            (1, -2, -3),
            (1, 2, -3)
        }

        sampler = dataset.UnigenDatasetGenerator()
        # 5 is the max number of negatives, with 20 it will reach max attempts
        negatives = sampler.generate_negative_samples(f, 20, max_attempts=200000)
        self.assertEqual(5, len(negatives))

        # checks if all negatives are unique: transform into set and see if the length did not reduce
        neg_set = set(tuple(neg) for neg in negatives)
        self.assertEqual(5, len(neg_set))

        # checks if the returned set of negatives is correct
        self.assertEqual(expected_negatives, neg_set)


if __name__ == '__main__':
    unittest.main()
