import os

import fire
import main
import tarfile


def run(instances, output='out.csv', extraction_point='/tmp/satinstances',
        cvfolds=5, model='MLP',
        mlp_layers=[200, 100], mlp_activation='relu',
        solver='unigen', save_dataset=False):
    """
    Extracts all files in a tar.gz file and runs the experiment for each one of them
    :param solver: name of the SAT solver to find the satisfying samples
    :param instances: .tar.gz file containing the .cnf instances
    :param cvfolds: number of folds for cross-validation
    :param model: learner (MLP, DecisionTree or RF for random forest)
    :param mlp_layers: list with #neurons in each hidden layer (from command line, pass it without spaces e.g. [200,100,50])
    :param mlp_activation: MLP's activation function
    :param output: path to write results to (csv format)
    :param extraction_point: point to extract the cnf instances
    :param save_dataset: whether to save the dataset generated from the cnf files
    :return:
    """
    # creates the extraction point if it does not exist
    os.makedirs(extraction_point, exist_ok=True)

    # extracts the instances
    print(f'Extracting contents of {instances} to {extraction_point}')
    with tarfile.open(instances) as tf:
        tf.extractall(extraction_point)

    # run each instance in the extraction point (finds all files there recursively)
    for root, dirs, files in os.walk(extraction_point):
        print(f'{len(files)} files are at {root}.')

        for f in files:
            print(f'Running {f}...')
            # runs the experiment on formula f, all parameters received from cmdline are passed
            main.evaluate(os.path.join(root, f), output=output,
                          cvfolds=cvfolds, model=model,
                          mlp_layers=mlp_layers, mlp_activation=mlp_activation,
                          solver=solver, save_dataset=save_dataset)
            print()  # just a newline
    print(f"Finished all instances. Extracted instances and datasets are at {extraction_point}.")
    # shutil.rmtree(extraction_point)
    # print('Done')


if __name__ == '__main__':
    fire.Fire(run)
