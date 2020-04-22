# MLSat
Machine Learning on SATisfiability

Currently a working prototype that implements part of Moshe Vardi's idea (e.g. it does a different approach on 2):

1. Start with a large Boolean formula f (say, industrial SAT benchmark).
2. Generate many samples satisfying f or \neg f.
(Using, for example, [https://www.nature.com/articles/s42256-018-0002-3](https://www.nature.com/articles/s42256-018-0002-3))
3. Train a DNN on this labeled sample.
4. Check whether it is a good apprcximation of f.

## Installation

You need the following libraries (commands to install assume a conda environment):

* scikit-learn (`conda install -c anaconda scikit-learn`)
* pandas (` conda install -c anaconda pandas`)
* fire (`conda install -c conda-forge fire`)
* pysat (`pip install python-sat[pblib,aiger]`)

Then you just need to clone this repository:

`git clone https://github.com/andertavares/mlsat.git` and enter the new directory `mlsat` to be able to execute.

## Execution

`python main.py cnf=path_to_cnf_file`

If cnf is omitted, it runs with  `instances/bw_large.d.cnf` by default. This is a blocks world instance from satlib (https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/PLANNING/BlocksWorld/descr.html)


