import random
import sys

import pandas as pd
from pysat.formula import CNF
from pysat.solvers import Glucose3, Solver, NoSuchSolverError


def generate_dataset(cnf_file, solver_name='Glucose3', max_samples=1000):
    """
    Generates a dataset from the boolean formula in the informed CNF file.
    Particularly, it enumerates all satisfying samples with a SAT solver.
    Then it generates the same number of unsat samples.
    Unsat samples are generated by flipping one bit of a sat sample.

    All samples are shuffled and assembled into two dataframes.
    The first dataframe contains the binary (0/1) inputs.
    The second dataframe contains the corresponding binary label (0/1) for unsat/sat,
    respectively.

    :param cnf_file: path to the file in CNF (Dimacs) format (see https://people.sc.fsu.edu/~jburkardt/data/cnf/cnf.html)
    :param solver_name: name of the solver to generate the positive instances
    :param max_samples: maximum number of samples, half of them will be positive, half negative
    :return: tuple with 2 dataframes: inputs and labels
    """
    
    print(f'Reading boolean formula from {cnf_file}.')
    formula = CNF(from_file=cnf_file)

    # num_vars = formula.nv
    unsat_list = []

    print(f'Finding satisfiable assignments with {solver_name}')
    try:
        with Solver(name=solver_name, bootstrap_with=formula) as solver:

            # for each positive (sat) instance, flip a literal to generate a negative (unsat) instance

            # generates the satisfying instances by querying the solver
            sat_list = []
            for i, solution in enumerate(solver.enum_models()):
                sat_list.append(solution)
                if i+1 >= max_samples / 2:  # adds 1 to index as it starts at zero
                    print(f"Limit number of {max_samples/2} positive samples reached.")
                    break

            print(f'Found {len(sat_list)} sat instances.')
            if len(sat_list) == 0:
                print('WARNING: No sat instance found, returning empty dataset.')
                return [], []

            print('Generating the same number of unsat instances.')

            # transforming each sat instance into tuple eases the generation of negative instances
            sat_set = set([tuple(s) for s in sat_list])  # set is much quicker to verify an existing instance

            for assignment in sat_list:
                for i, literal in enumerate(assignment):
                    tentative_unsat = list(assignment)
                    tentative_unsat[i] = -tentative_unsat[i]  # negating one literal
                    if tuple(tentative_unsat) not in sat_set:
                        unsat_list.append(tentative_unsat)
                        break  # goes on to next assignment
                    # print(f'negated {i}-th')
    except NoSuchSolverError as e:
        print(f'ERROR: no solver named "{solver_name}" was found. '
              f'Please use one of the names in '
              f'https://pysathq.github.io/docs/html/api/solvers.html#pysat.solvers.SolverNames'
              f', such as Glucose3, for example. Exiting.')
        sys.exit(0)

    # uncomment below to test duplicates
    # sat_list.append(sat_list[0])

    # appends the labels (1 to sat samples, 0 to unsat samples)
    for sat in sat_list:
        sat.append(1)
    for unsat in unsat_list:
        unsat.append(0)

    print(f'Preparing dataset.')

    # concats and shuffles the two lists
    all_data = sat_list + unsat_list
    # random.seed(2) # uncomment to debug (otherwise each shuffle will give a different array)
    random.shuffle(all_data)

    # column names = [x1, x2, ..., xn, f] (each x_i is a variable and f is the label)
    input_names = [f'x{i}' for i in range(1, len(all_data[0]))]
    df = pd.DataFrame(all_data, columns=input_names + ['f'])
    if any(df.duplicated(input_names)):
        print('ERROR: there are duplicate inputs in the dataset. Returning empty.')
        return [], []

    # replaces negatives by 0 and positives by 1
    df.mask(df < 0, 0, inplace=True)
    df.mask(df > 0, 1, inplace=True)

    # breaks into input features & label
    data_x = df.drop('f', axis=1)
    data_y = df['f']

    return data_x, data_y
