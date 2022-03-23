import os
import fire
import tarfile

import learn_valiant


def run_learn(instances, output='out.csv', extraction_path='/tmp/learn_instances',
              cvfolds=5, arity=3, debug=False):
    """
    Extracts all files in a tar.gz file and runs the experiment for each one of them
    :param instances: .tar.gz file containing the .cnf instances
    :param cvfolds: number of folds for cross-validation
    :param output: path to write results to (csv format)
    :param extraction_path: point to extract the cnf.pkl.gz instances
    :param arity: arity of CNF formulas
    :param debug: run with verbose output steps
    :return:
    """

    os.makedirs(extraction_path, exist_ok=True)

    # extracts the instances
    print(f'Extracting contents of {instances} to {extraction_path}')
    with tarfile.open(instances) as tf:
        tf.extractall(extraction_path)

    for root, dirs, files in os.walk(extraction_path):
        print(f'{len(files)} files are at {root}.')

        for f in files:
            print(f'Running {f}...')
            # runs the experiment on formula f, all parameters received from cmdline are passed
            learn_valiant.evaluate(os.path.join(root, f),
                                   output=output,
                                   cvfolds=cvfolds,
                                   cnf_arity=arity,
                                   debug=debug)
            print()  # just a newline
    print(f"Finished all instances. Extracted instances and datasets are at {extraction_path}.")


if __name__ == '__main__':
    fire.Fire(run_learn)
