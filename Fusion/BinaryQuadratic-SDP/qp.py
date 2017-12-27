# mosek.com
#
# Example for solving binary quadratic problems
# with Branch and Bound
#
# See the accompanying notebook for description and comments

import branchbound
from branchbound import BB 

import concurrent.futures as con
import sys, time, os
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Loads an example in biqmac format
def biqmacExample(fname):
    with open(fname) as f:
        cont = f.readlines()
    n, m = [int(x) for x in cont[0].split(' ')]
    Q = np.zeros([n,n])
    for i in range(1, m+1):
        a,b,v = cont[i].split(' ') 
        a,b,v = (int(a),int(b),int(v))
        Q[a-1][b-1] = Q[b-1][a-1] = v
    return Q, np.zeros(n, dtype=float), 0.0

# Loads a random example
def randomExample(n, seed=None):
    if seed:
        np.random.seed(seed)
    Q1 = np.random.normal(0.0, 1.0, (n,n))
    return (Q1+Q1.transpose())/2, np.zeros(n, dtype=float), 0.0

# Plotting
def makePlot(res, idx, name):
    plt.figure(idx)
    plt.yscale('log')
    plt.scatter([r[0] for r in res], [r[idx] for r in res], color='blue')
    plt.savefig(name+'.png')

# Run the solver
def Solve(solver):
    try:
        solver.solve()
        solver.summary(verbose=True)    
        return [solver.n, solver.relSolved, solver.intpntIter, solver.optTime]
    except Exception as e:
        print e

# Run lots of biqmac tests and show results
def testBiq(maxN, workers=1):
    # Find all files
    fnamelist = []
    for root, dirs, files in os.walk("./biq"):
        for fname in files:
            if ".sparse" in fname:
                fnamelist.append(os.path.join(root, fname))

    # Run examples and collect statistics
    res = []
    count = 0
    jobs = []
    with con.ProcessPoolExecutor(max_workers=workers) as executor:    
        for fname in fnamelist:
            Q, P, R = biqmacExample(fname)
            n = len(Q)
            if n <= maxN:
                count += 1
                solver = BB(Q, P, R, verbose=False) 
                jobs.append(executor.submit(Solve, solver))

        con.wait(jobs)
        res = [job.result() for job in jobs]

    # Simple plot
    if res:
        makePlot(res, 1, 'relaxations-biq')
        makePlot(res, 2, 'iterations-biq')
        makePlot(res, 3, 'optTime-biq')

# Run lots of random tests and show results
def testRandom(nlist, workers=1):
    # Run examples and collect statistics
    res = []
    count = 0
    jobs = []
    with con.ProcessPoolExecutor(max_workers=workers) as executor:    
        for n in nlist:
            Q, P, R = randomExample(n)
            count += 1
            solver = BB(Q, P, R, verbose=False) 
            jobs.append(executor.submit(Solve, solver))

    con.wait(jobs)
    res = [job.result() for job in jobs]

    # Simple plot
    if res:
        makePlot(res, 1, 'relaxations-rnd')
        makePlot(res, 2, 'iterations-rnd')
        makePlot(res, 3, 'optTime-rnd')

np.random.seed(1123)
#testRandom([30,40,50,60,70,80,90,100]*10,44)
testBiq(100, 44)