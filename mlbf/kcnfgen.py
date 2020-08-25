import os
import subprocess

import fire


def phase_transition(num_vars):
    phases = {  # vars->clauses -- hard-coded values based on previous research
        20: 91,
        50: 218,
        100: 431,
        150: 645,
        200: 854
    }

    if num_vars in phases:
        return phases[num_vars]

    # Crawford 1996's equation
    return int(4.258*num_vars + 58.26 * num_vars**(-2/3))


def kcnfgen(output, n, m, k=3):
    """
    Generates a random k-CNF with the specified parameters.
    This is just a wrapper on cnfgen randkcnf passing the proper parameters.
    :param output: prefix of where the files will be saved
    :param n: number of variables
    :param m: number of clauses
    :param k: number of literals per clause
    """
    #for i in range(num_formulas):
        # output = f'{dest_prefix}/{i}_{k}_{n}_{m}.cnf'
    subprocess.call(['cnfgen', '--output', output, 'randkcnf', str(k), str(n), str(m)])


def generate_instances(prefix, n, m, k=3, num_instances=100, tmpfile='/tmp/candidate.cnf'):
    num_sat, num_unsat = 0, 0
    for i in range(num_instances):
        print('calling cnfgen')
        kcnfgen(tmpfile, n, m, k)

        # TODO collect statistics

        try:  # checks satisfiability with minisat
            print('calling minisat')
            subprocess.check_call(['minisat', tmpfile], stderr=subprocess.STDOUT)
        except subprocess.CalledProcessError as ex:
            if ex.returncode == 10:  # satisfiable
                os.rename(tmpfile, f'{prefix}/sat_{num_sat:05}_{k}_{n}_{m}.cnf')
                num_sat += 1
            else:  # unsatisfiable
                os.rename(tmpfile, f'{prefix}/sat_{num_unsat:05}_{k}_{n}_{m}.cnf')
                num_unsat += 1


if __name__ == '__main__':
    fire.Fire()
