'''
File disks.py
Copyright: Mosek ApS
Content: Geometric facility location - planar disk coverage problems. Mixed integer conic program.
         Accompanies an online notebook.
'''

from mosek.fusion import *
import numpy as np
import sys

def basicModel(k, P, bigM):
	n, dim = len(P), len(P[0])
	M = Model("disks")
	R = M.variable(k, Domain.greaterThan(0.0))
	X = M.variable([k,dim], Domain.unbounded())
	S = M.variable([n,k], Domain.binary())

	RRep = Var.repeat(R, n)
	Penalty = Expr.flatten(Expr.mul(bigM, Expr.sub(1.0, S)))
	CoordDiff = Expr.sub(np.repeat(P,k,0), Var.repeat(X, n))

	M.constraint(Expr.hstack(Expr.add(RRep,Penalty), CoordDiff), Domain.inQCone())

	return M, R, X, S

def bigM(P):
	return 2*np.shape(P)[1]*max(np.amax(P,0)-np.amin(P,0))

def minimalEnclosing(P, dump=""):
	M, R, X, S = basicModel(1, P, 0.0)
	M.constraint(S, Domain.equalsTo(1.0))
	M.objective(ObjectiveSense.Minimize, R.index(0))
	M.solve()
	return R.level(), X.level()

def maxCoverage(P, rMax, dump=""):
	M, R, X, S = basicModel(1, P, bigM(P))
	M.constraint(R, Domain.equalsTo(rMax))
	M.objective(ObjectiveSense.Maximize, Expr.sum(S))
	M.solve()
	return R.level(), X.level()

def minDiamKCircleCover(P, k, dump=""):
	M, R, X, S = basicModel(k, P, bigM(P))
	M.constraint(Expr.sum(S,1), Domain.greaterThan(1.0))
	M.objective(ObjectiveSense.Minimize, Expr.sum(R))
	M.solve()
	return R.level(), X.level()

def minAreaKCircleCover(P, k, dump=""):
	M, R, X, S = basicModel(k, P, bigM(P))
	t = M.variable(1, Domain.greaterThan(0.0))
	M.constraint(Expr.vstack(t, R), Domain.inQCone())
	M.constraint(Expr.sum(S,1), Domain.greaterThan(1.0))
	M.objective(ObjectiveSense.Minimize, t)
	if dump:
		M.writeTask("diskcover"+dump+".mps.gz")
		return
	else:
		M.solve()
		return R.level(), X.level()
	return R.level(), X.level()

def maxKCoverage(P, k, rMax, dump=""):
	M, R, X, S = basicModel(k, P, bigM(P))
	M.constraint(R, Domain.equalsTo(rMax))
	t = M.variable("t", len(P), Domain.binary())
	M.constraint(Expr.sub(t, Expr.sum(S,1)), Domain.lessThan(0.0)) 
	M.objective(ObjectiveSense.Maximize, Expr.sum(t))
	if dump:
		M.writeTask("diskcover"+dump+".mps.gz")
		return
	else:
		M.solve()
		return R.level(), X.level()

def display(P, R, X):
	try:
		import matplotlib.pyplot as plt
		plt.scatter(*zip(*P))
		plt.axis("equal")
		for i in range(len(R)):
			c = plt.Circle((X[2*i],X[2*i+1]), R[i], fc="r", color="r", alpha=0.5)
			plt.gcf().gca().add_artist(c)
			plt.gcf().gca().plot(X[2*i],X[2*i+1],"or")
		plt.show()
	except:
		pass

def data(n):
    np.random.seed(1236)
    return [[x,y] for x,y in np.broadcast(np.random.uniform(0.0, 1.0, n), np.random.uniform(0.0, 0.3, n))]

def gaussianData(n):
    np.random.seed(1236)
    return [[x,y] for x,y in np.broadcast(np.random.normal(0.0, 1.0, n), np.random.normal(0.0, 1.0, n))]

############################################################

P=data(20); r, x = maxKCoverage(P, 3, 0.1); display(P, r, x)


#Generating test cases as mps.gz
P=data(7); maxKCoverage(P, 3, 0.1, "_1"); 
P=data(12); maxKCoverage(P, 3, 0.1, "_2"); 
P=data(20); maxKCoverage(P, 3, 0.1, "_3"); 
P=gaussianData(7); maxKCoverage(P, 3, 0.1, "_4"); 
P=gaussianData(12); maxKCoverage(P, 3, 0.1, "_5"); 
P=gaussianData(20); maxKCoverage(P, 3, 0.1, "_6"); 
P=data(7); minAreaKCircleCover(P, 3, "_7");
P=data(12); minAreaKCircleCover(P, 3, "_8");
P=data(20); minAreaKCircleCover(P, 3, "_9");
P=data(30); minAreaKCircleCover(P, 4, "_10");


