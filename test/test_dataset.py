import os
import sys
import unittest

from pysat.formula import CNF

# not proud of this hack but I just can't manage the imports work as intended without it
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'mlsat'))
import mlsat.positives
import mlsat.negatives
import mlsat.dataset as dataset


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
        data = mlsat.dataset.prepare_dataset(positives, negatives)
        data_x, data_y = mlsat.dataset.get_xy_data(data)
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

    def test_negative_simple(self):
        cnf_file = '/tmp/simple_negative.cnf'
        f = CNF(from_clauses=[[-1, 2], [3]])
        expected_negatives = {  # set of tuples, contains all unsat assignments for the formula
            (1, -2, 3),
            (-1, -2, -3),
            (-1, 2, -3),
            (1, -2, -3),
            (1, 2, -3)
        }
        f.to_file(cnf_file)

        negatives = mlsat.negatives.uniformly_negative_samples(cnf_file, 5)  # 5 is the max number of negatives
        self.assertEqual(5, len(negatives))

        # checks if all negatives are unique: transform into set and see if the length did not reduce
        neg_set = set(tuple(neg) for neg in negatives)
        self.assertEqual(5, len(neg_set))

        # checks if the returned set of negatives is correct
        self.assertEqual(expected_negatives, neg_set)
        os.unlink(cnf_file)

    def test_negative_max_attempts(self):
        cnf_file = '/tmp/test_max_attempts.cnf'
        f = CNF(from_clauses=[[-1, 2], [3]])
        expected_negatives = {  # set of tuples, contains all unsat assignments for the formula
            (1, -2, 3),
            (-1, -2, -3),
            (-1, 2, -3),
            (1, -2, -3),
            (1, 2, -3)
        }
        f.to_file(cnf_file)

        # 5 is the max number of negatives, with 20 it will reach max attempts
        negatives = mlsat.negatives.uniformly_negative_samples(cnf_file, 20, max_attempts=200000)
        self.assertEqual(5, len(negatives))

        # checks if all negatives are unique: transform into set and see if the length did not reduce
        neg_set = set(tuple(neg) for neg in negatives)
        self.assertEqual(5, len(neg_set))

        # checks if the returned set of negatives is correct
        self.assertEqual(expected_negatives, neg_set)

        # removes the temporary file
        os.unlink(cnf_file)

    def test_unigen_retrieve_samples(self):
        """
        Emulates a file created by unigen and tests whether
        the samples there are correctly retrieved
        :return:
        """

        # suppose this formula: f = CNF(from_clauses=[[-1, 2], [3]])
        # here are the 3 positive samples:
        expected_positives = { # set of tuples
            (-1, 2, 3),
            (-1, -2, 3),
            (1, 2, 3),
        }

        # I'll save it in a temp file with the Unigen2 format: v1 2 ... N 0:M
        with open('/tmp/retrieval_test.txt', 'w') as sample_file:
            for pos in expected_positives:
                sample_file.write(f"v{' '.join([str(lit) for lit in pos])} 0:1\n")

        sampler = mlsat.positives.UnigenSampler()
        retrieved = sampler.retrieve_samples('/tmp/retrieval_test.txt')
        # transforms the list of lists in a set of tuples for comparison
        retrieved_set = set([tuple([lit for lit in pos]) for pos in retrieved])

        self.assertEqual(expected_positives, retrieved_set)

    def test_unigen_generate_dataset_small_formula(self):
        cnf_file = '/tmp/test_small.cnf'
        f = CNF(from_clauses=[[1, 2, 3], [4]])
        '''
        the formula above has 7 positive samples:
        positives = {  
            (1, 2, 3, 4),
            (1, 2, -3, 4),
            (1, -2, 3, 4),
            (1, -2, -3, 4),
            (-1, 2, 3, 4),
            (-1, 2, -3, 4),
            (-1, -2, 3, 4),
        }
        '''
        f.to_file(cnf_file)

        sampler = mlsat.positives.UnigenSampler()

        # a formula with very few solutions will return an empty dataset
        data = sampler.sample(cnf_file, 50)
        self.assertEqual(0, len(data))

        # deletes the temp file used to store the formula
        os.unlink(cnf_file)

    def test_unigen_generate_dataset(self):
        cnf_file = '/tmp/test.cnf'
        f = CNF(from_clauses=[[1, 2, 3, 4], [5, 6]])
        # the formula above has 45 positive & 19 negative samples
        # positives enumerated below (with Glucose3)
        positives = {
            (1, -2, -3, -4, 5, -6),
            (1, 2, -3, -4, 5, -6),
            (1, 2, 3, -4, 5, -6),
            (1, 2, 3, 4, 5, -6),
            (1, 2, 3, 4, 5, 6),
            (-1, 2, 3, 4, 5, 6),
            (-1, -2, 3, 4, 5, 6),
            (-1, -2, -3, 4, 5, 6),
            (-1, -2, 3, -4, 5, 6),
            (-1, -2, 3, -4, -5, 6),
            (-1, -2, 3, -4, 5, -6),
            (1, -2, 3, -4, 5, -6),
            (-1, 2, 3, -4, 5, -6),
            (-1, 2, -3, -4, 5, -6),
            (-1, 2, -3, 4, 5, -6),
            (-1, 2, -3, 4, 5, 6),
            (-1, 2, -3, -4, 5, 6),
            (1, 2, -3, -4, 5, 6),
            (1, -2, -3, -4, 5, 6),
            (1, -2, 3, -4, 5, 6),
            (1, -2, 3, -4, -5, 6),
            (1, -2, 3, 4, 5, 6),
            (1, -2, 3, 4, -5, 6),
            (1, -2, -3, 4, 5, 6),
            (1, -2, -3, 4, -5, 6),
            (1, -2, -3, -4, -5, 6),
            (1, 2, -3, -4, -5, 6),
            (1, 2, 3, -4, 5, 6),
            (1, 2, 3, -4, -5, 6),
            (1, 2, 3, 4, -5, 6),
            (1, 2, -3, 4, 5, 6),
            (1, 2, -3, 4, -5, 6),
            (-1, 2, -3, 4, -5, 6),
            (-1, 2, -3, -4, -5, 6),
            (-1, 2, 3, -4, 5, 6),
            (-1, 2, 3, -4, -5, 6),
            (-1, 2, 3, 4, -5, 6),
            (-1, -2, 3, 4, -5, 6),
            (-1, -2, -3, 4, -5, 6),
            (1, 2, -3, 4, 5, -6),
            (1, -2, -3, 4, 5, -6),
            (-1, -2, -3, 4, 5, -6),
            (-1, -2, 3, 4, 5, -6),
            (-1, 2, 3, 4, 5, -6),
            (1, -2, 3, 4, 5, -6)
        }

        # negatives enumerated below (by hand)
        negatives = {
            (-1, -2, -3, -4, -5, -6),  # negate first clause
            (-1, -2, -3, -4, -5, 6),
            (-1, -2, -3, -4, 5, -6),
            (-1, -2, -3, -4, 5, 6),

            (-1, -2, -3, 4, -5, -6),  # negate 2nd clause, 4th lit up
            (-1, -2, 3, -4, -5, -6),  # negate 2nd clause, 3rd lit up
            (-1, -2, 3, 4, -5, -6),   # ...

            (-1, 2, -3, -4, -5, -6),  # negate 2nd clause, 2nd lit up
            (-1, 2, -3, 4, -5, -6),  # ...
            (-1, 2, 3, -4, -5, -6),  # ...
            (-1, 2, 3, 4, -5, -6),  # ...

            (1, -2, -3, -4, -5, -6),
            (1, -2, -3, 4, -5, -6),  # negate 2nd clause, 1st lit up
            (1, -2, 3, -4, -5, -6),
            (1, -2, 3, 4, -5, -6),
            (1, 2, -3, -4, -5, -6),
            (1, 2, -3, 4, -5, -6),
            (1, 2, 3, -4, -5, -6),
            (1, 2, 3, 4, -5, -6),
        }
        f.to_file(cnf_file)

        sampler = mlsat.positives.UnigenSampler()
        sampled_positives = sampler.sample(cnf_file, 500)
        sampled_negatives = mlsat.negatives.uniformly_negative_samples(cnf_file, 500)

        data_x, data_y = dataset.get_xy_data(dataset.prepare_dataset(sampled_positives, sampled_negatives))

        # I expect the dataset to contain all 64 possible assignments
        self.assertEqual(64, len(data_x))
        self.assertEqual(64, len(data_y))

        # create a new dataset with data_x and y appended column-wise
        df = data_x.copy()
        df['y'] = data_y

        # convert expected to 0/1 instead of n/-n
        binary_pos = {  # +[1]  because each one is a positive example
            tuple([0 if lit < 0 else 1 for lit in assignment] + [1]) for assignment in positives
        }
        binary_neg = {  # +[0]  because each one is a positive example
            tuple([0 if lit < 0 else 1 for lit in assignment] + [0]) for assignment in negatives
        }

        # creates a set out of the dataset and compare it with the expected one
        dataset_set = set([tuple(row) for row in df.values])
        self.assertEqual(binary_pos.union(binary_neg), dataset_set)

        # deletes the temp file used to store the formula
        os.unlink(cnf_file)


if __name__ == '__main__':
    unittest.main()
