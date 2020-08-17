import datetime
import math
import os
import fire

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate

import dataset
import numpy as np


def mlpsize(*inputs, solver='unigen', output='out.csv', cvfolds=5,
            max_neurons=512, mlp_activation='relu', metric='accuracy', save_dataset=True):
    """
    Tests various hidden-layer sizes until it goes perfect on a given formula, or formulas.
    (much code's been shamelessly copied from main.py
    TODO: modularize the code)

    :param inputs: path to (multiple) dataset (.pkl.gz) or CNF (Dimacs) files (see https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html)
    :param solver: name of the SAT solver to sample satisfying samples
    :param output: path to output file
    :param cvfolds: number of folds for cross-validation
    :param max_neurons: max # neurons
    :param mlp_activation: MLP's activation function
    :param save_dataset: if True, saves the dataset as 'input.cnf.pkl.gz'
    :param metric: score metric
    :return:
    """
    for formula in inputs:
        start = datetime.datetime.now()

        data_x, data_y = dataset.get_dataset(formula, solver)

        if len(data_x) < 100:
            print(f'{inputs} has {len(data_x)} instances, which is less than 100 (too few to learn). Aborting.')
            return

        write_header(output)

        # goes from 1, 2, 4, ..., max_neurons until it finds the optimal accuracy
        for i in range(int(math.log2(max_neurons)) + 1):
            num_neurons = 2 ** i  # the next power of 2
            learner = MLPClassifier(hidden_layer_sizes=(num_neurons,), activation=mlp_activation)
            scores = cross_validate(learner, data_x, data_y, cv=cvfolds, scoring=metric)

            with open(output, 'a') as outstream:
                # gathers accuracy and precision by running the model and writes it to output
                print(num_neurons, scores)
                score, std = np.mean(scores['test_score']), np.std(scores['test_score'])
                finish = datetime.datetime.now()
                outstream.write(f'{os.path.basename(formula)},{solver},{mlp_activation},{num_neurons},{cvfolds},{metric},{score},{std},{start},{finish}\n')
                if score == 1 and std == 0:
                    print(f'Perfect {metric} for {formula} with {num_neurons} neurons in 1 hidden layer')
                    break


def write_header(output):
    """
    Creates the header in the output file if it does not exist
    :param output: path to the output file
    :return:
    """
    if output is not None and not os.path.exists(output):
        with open(output, 'w') as out:
            out.write('formula,sampler,activation,#neurons,cvfolds,metric,mean,std,start,finish\n')


if __name__ == '__main__':
    fire.Fire(mlpsize)
