# mosek.com
#
# Example for solving binary quadratic problems
#
# See the accompanying notebook for description and comments

import branchbound
from branchbound import BB 

import numpy as np

n=25
Q1 = np.random.normal(0.0, 1.0, (n,n))
solver = BB((Q1+Q1.transpose())/2, np.random.uniform(-1.0, 3.0, n), 0.0)
solver.solve()
solver.summary()