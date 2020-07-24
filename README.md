# MLBF
Machine Learning on Boolean Formulas

Currently a working prototype that implements part of Moshe Vardi's idea (e.g. it does a different approach on 2):

1. Start with a large Boolean formula f (say, industrial SAT benchmark).
2. Generate many samples satisfying f or \neg f.
(Using, for example, Unigen)
3. Train a DNN on this labeled sample.
4. Check whether it is a good apprcximation of f.

## Installation

You need the following libraries (commands to install assume a conda environment):

* scikit-learn & pandas (`conda install -c anaconda scikit-learn pandas`)
* fire (`conda install -c conda-forge fire`)
* pysat (`pip install python-sat[pblib,aiger]`)

Then you just need to clone this repository:

`git clone https://github.com/andertavares/mlbf.git` and enter the new directory `mlsat` to be able to execute.

## Execution
__[outdated i.e. not valid for the current version!]__

* Single instance: 
`python main.py cnf=path_to_cnf_file --solver=SolverName --output=output_file`

Defaults are: `cnf=instances/bw_large.d.cnf`, `solver=Glucose3`, `output=out.csv`. The instance is one from blocks world from satlib (https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/PLANNING/BlocksWorld/descr.html). The solver must be one of: https://pysathq.github.io/docs/html/api/solvers.html.

* Many instances in a .tar.gz file:
This is particularly useful to evaluate in SATlib instances, which have many related intances packed in a .tar.gz file. The command is (values assigned to [optional] parameters are the default ones, which can be replaced as you wish):

`python run_instances.py instances_file [--output_file=out.csv extraction_point=/tmp/satinstances solver='Glucose3']`


