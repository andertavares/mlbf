import pandas as pd
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


if __name__ == '__main__':
    unittest.main()
