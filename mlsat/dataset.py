import os
import random

import fire
import numpy as np
import pandas as pd

from positives import PySATSampler, UnigenSampler
import negatives as negative_sampler


def prepare_dataset(positives, negatives):
    """
    Creates the dataset from the positive and negative samples.
    Adds the labels, concatenates, shuffles, creates a
    dataframe and separate into inputs and labels
    :param positives:list
    :param negatives:list
    :return: pandas.DataFrame object
    """
    print(f'Preparing dataset.')

    # checks the validity of the samples
    if len(positives) == 0:
        print("No positives samples. Returning empty dataset")
        return [], []
    if len(negatives) == 0:
        print("No negative samples. Returning empty dataset")
        return [], []

    # appends the labels (1 to sat samples, 0 to unsat samples)
    for p in positives:
        p.append(1)
    for n in negatives:
        n.append(0)

    # concats the two lists and shuffles
    all_data = positives + negatives
    # random.seed(2) # uncomment to debug (otherwise each shuffle will give a different array)
    random.shuffle(all_data)

    # column names = [x1, x2, ..., xn, f] (each x_i is a variable and f is the label)
    input_names = [f'x{i}' for i in range(1, len(all_data[0]))]
    df = pd.DataFrame(all_data, columns=input_names + ['f'])
    if any(df.duplicated(input_names)):
        print('ERROR: there are duplicate inputs in the dataset. Returning empty.')
        return [], []

    # replaces negated by 0 and asserted by 1
    df.mask(df < 0, 0, inplace=True)
    df.mask(df > 0, 1, inplace=True)

    if df.isin([np.nan, np.inf, -np.inf]).values.any():
        print("ERROR: there are invalid samples in the dataset. Returning empty.")
        return [], []

    return df


def generate_dataset(cnf, solver='unigen', num_positives=500, num_negatives=500, save_dataset=True, overwrite=False):
    """
    Generates a dataset out of a CNF boolean formula
    :param cnf: path to the boolean formula in DIMACS CNF format
    :param solver: unigen or the name of a PySAT solver
    :param num_positives: number of positive samples
    :param num_negatives: number of negative samples
    :param save_dataset: if True, saves the dataset as cnf_solver_pos_neg.pkl.gz, where pos & neg are the actual number of samples
    :param overwrite: if True, overwrites an existing datset
    :return: two dataframes with the input data & labels (X and y in ML library parlance)
    """

    positive_sampler = UnigenSampler() if solver == 'unigen' else PySATSampler(solver_name=solver)
    positives = positive_sampler.sample(cnf, num_positives)
    negatives = negative_sampler.uniformly_negative_samples(cnf, num_negatives)
    df = prepare_dataset(positives, negatives)

    print(f'{len(df)} instances generated for {cnf}')
    # FIXME do not save if there are no instances
    if save_dataset:
        dataset_output = f'{cnf}_{solver}_{len(positives)}_{len(negatives)}.pkl.gz'
        if not overwrite and os.path.exists(dataset_output):
            print(f'Output "{dataset_output}" already exists, will not save.')
        else:
            print(f'Saving dataset to {dataset_output}.')
            df.to_pickle(dataset_output, compression='gzip')

    # breaks into input features & label
    data_x = df.drop('f', axis=1)
    data_y = df['f']

    return data_x, data_y


def cli_generate_dataset(cnf, solver='unigen', num_positives=500, num_negatives=500, save_dataset=True, overwrite=False):
    """
    This function just calls 'generate_dataset' but does not return data for cleaner use with fire.Fire
     :param cnf: path to the boolean formula in DIMACS CNF format
    :param solver: unigen or the name of a PySAT solver
    :param num_positives: number of positive samples
    :param num_negatives: number of negative samples
    :param save_dataset: if True, saves the dataset as cnf_solver_pos_neg.pkl.gz, where pos & neg are the actual number of samples
    :param overwrite: if True, overwrites an existing datset
    :return:
    """
    generate_dataset(cnf, solver, num_positives, num_negatives, save_dataset, overwrite)


if __name__ == '__main__':
    fire.Fire(cli_generate_dataset)
