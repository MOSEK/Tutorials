# mosek.com
#
# Example for solving Binary Least Squares
#
# See the accompanying notebook for description and comments

import branchbound
from branchbound import BB 

from mosek.fusion import *

import concurrent.futures as con
import sys, time, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loads a random example
def randomExample(n, m, seed=None):
    if seed:
        np.random.seed(seed)
    A = np.random.normal(0.0, 1.0, (m,n))
    c = np.random.uniform(0.0, 1.0, n)
    b = A.dot(c)
    return A, b

# Binary least squares - direct Fusion MIP model
def blsMIP(A, b):
    n = len(A[0])
    M = Model()
    x = M.variable("x", n, Domain.binary())
    t = M.variable("t")
    M.constraint(Expr.flatten(Expr.vstack(t, Expr.sub(Expr.mul(A,x),b))), Domain.inQCone())
    M.objective(ObjectiveSense.Minimize, t)
    return M, x

# Plotting
def makePlot(res, idx1, idx2, name):
    plt.figure(idx1)
    plt.yscale('log')
    plt.scatter([r[0] for r in res], [r[idx1] for r in res], color='blue')
    plt.scatter([r[0] + 0.5 for r in res], [r[idx2] for r in res], color='red')
    plt.savefig(name+'.png')

# Run the solver and compare it with MIP solver
def Solve(solver, A, b, m):
    try:
        solver.solve()
        #solver.summary(verbose=True)  

        M, x = blsMIP(A, b)
        M.setSolverParam("numThreads", 1)
        M.solve()

        if not np.all(x.level() == solver.xx):
            print("Obtained different solutions {0} {1} {2}".format(m, solver.ub, M.primalObjValue()**2)) 
        return [m, 
                solver.relSolved, 
                solver.intpntIter, 
                solver.optTime,
                M.getSolverIntInfo("mioNumRelax"),
                M.getSolverLIntInfo("mioIntpntIter"),
                M.getSolverDoubleInfo("optimizerTime")]
    except Exception as e:
        print e
    finally:
        M.dispose()

# Run lots of random tests and show results
def testRandom(n, mlist, workers=1):
    # Run examples and collect statistics
    res = []
    jobs = []
    with con.ProcessPoolExecutor(max_workers=workers) as executor:    
        for m in mlist:
            A, b = randomExample(n, m)
            Q = A.transpose().dot(A)
            P = -2*b.transpose().dot(A)
            R = b.transpose().dot(b)

            solver = BB(Q, P, R, verbose=False) 

            jobs.append(executor.submit(Solve, solver, A, b, m))

    con.wait(jobs)
    res = [job.result() for job in jobs]

    print res

    # Simple plot
    if res:
        makePlot(res, 1, 4, 'relaxations-bls')
        makePlot(res, 2, 5, 'iterations-bls')
        makePlot(res, 3, 6, 'optTime-bls')

np.random.seed(1123)
n = 50
testRandom(n, [40,50,60,80,100,125,150]*8, 44)
