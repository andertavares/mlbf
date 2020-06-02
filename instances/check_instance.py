# very simple script that outputs #vars & #clauses for a given .cnf 
# useful to check if the instances correspond to their descriptions in a paper

from pysat.formula import CNF
import sys

f = CNF(sys.argv[1])
print(f'{sys.argv[1]} has {f.nv} vars and {len(f.clauses)} clauses')
