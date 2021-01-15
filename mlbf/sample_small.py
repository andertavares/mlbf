import os
import glob

import dataset
import fire


def sample_small(basedir, solver='Glucose3', *var_sizes):
    """
    Traverses the basedir, looking for instances without their corresponding dataset.
    Generates the dataset for these, using pysat-based sampler
    :param basedir: directory where the 'phase' dataset is located
    :param solver: sat solver used to enumerate instances
    :param var_sizes: will look for formuals with the number of variables in this list
    :return:
    """
    if len(var_sizes) == 0:
        var_sizes = range(10, 101, 10)

    for v in var_sizes:
        for cnf in glob.glob(f'{basedir}/v{v}/*/sat*.cnf'):
            if len(glob.glob(f'{cnf}*.pkl.gz')) == 0:
                print(f'Sampling for {cnf}')
                dataset.generate_dataset(cnf, solver)


if __name__ == '__main__':
    fire.Fire(sample_small)

# usage example:
# python mlbf/sample_small.py instances/phase Glucose3 10 20 30
