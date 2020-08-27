import os
import glob

import dataset
import fire


def sample_small(basedir, solver='Glucose3'):
    """
    Traverses the basedir, looking for instances without their corresponding dataset.
    Generates the dataset for these, using pysat-based sampler
    :param basedir:
    :param solver:
    :return:
    """
    var_sizes = range(10, 101, 10)  # [10,20,...,100]
    for v in var_sizes:
        for cnf in glob.glob(f'{basedir}/v{v}/*/sat*.cnf'):
            if len(glob.glob(f'{cnf}*.pkl.gz')) == 0:
                print(f'Sampling for {cnf}')
                dataset.generate_dataset(cnf, solver)


if __name__ == '__main__':
    fire.Fire(sample_small)