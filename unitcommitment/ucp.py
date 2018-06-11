'''
File ucm.py
Copyright: Mosek ApS
Content: A variant of the Unit Commitment Problem (CQ+MIP)
'''

from mosek.fusion import *
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Reading example files
import ucpParser

# A helper method repeating a vector in columns of a matrix r times (essentially hrepeat)
def repeat(v, R):
    return np.repeat(v, R).reshape(len(v), R)

'''
The model class. We use the following naming convention

Input data:

T               -   time frame
demand          -   demand for energy (for periods)
N               -   number of units
pmin,pmax       -   min/max energy production (for units)
rdown,rup       -   ramp down/up limit (for units)
mdown,mup       -   minimal down/up time (for units)
a,b,c           -   cost function coefficients (for units), cost of operating at power p is ap^2+bp+c
sc              -   startup cost (for units)
fx              -   fixed cost (for units) generated when the unit is operating
p0, u0, l0      -   initial state and how many time frames this state already existed (for units)

Decision variables:

p               -   production (units x frames)
u               -   0 unit inactive, 1 unit active (units x frames)
v               -   1 when unit starts up (units x frames)

The actual length of variables is T+1. The first column is reserved for input state.

'''

class UCP:
    # Initialize with input data
    def __init__(self, T, N):
        self.T, self.N = T, N
        self.M = Model()

        # Production level --- p_ih
        self.p = self.M.variable([self.N, self.T+1], Domain.greaterThan(0.0))   
        # Is active? --- u_ih
        self.u = self.M.variable([self.N, self.T+1], Domain.binary())   
        # Is at startup? --- v_ih >= u_ih-u_(i-1)h
        self.v = self.M.variable([self.N, self.T+1], Domain.binary())   
        self.M.constraint(Expr.sub(
                            self.v.slice([0,1],[N,T+1]), 
                            Expr.sub(self.u.slice([0,1], [N,T+1]), self.u.slice([0,0], [N,T]))), 
                          Domain.greaterThan(0.0))

# Production constraints
def consProd(ucp, pmin, pmax, demand):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    # Maximal and minimal production of each unit
    ucp.M.constraint(Expr.sub(p, Expr.mulElm(repeat(pmax, T+1), u)), Domain.lessThan(0.0))
    ucp.M.constraint(Expr.sub(p, Expr.mulElm(repeat(pmin, T+1), u)), Domain.greaterThan(0.0))
    # Total demand in periods is achieved
    ucp.M.constraint(Expr.sum(p, 0).slice(1,T+1), Domain.greaterThan(demand))

# Ramp constraints 
def consRamp(ucp, rdown, rup):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    Delta = Expr.sub(p.slice([0,1], [N,T+1]), p.slice([0,0], [N,T]))
    ucp.M.constraint(Delta, Domain.lessThan(repeat(rup, T)))
    ucp.M.constraint(Delta, Domain.greaterThan(-repeat(rdown, T)))

# Initial values
def consInit(ucp, p0, u0, l0):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    # Fix production in the immediately preceeding period
    ucp.M.constraint(p.slice([0,0],[N,1]), Domain.equalsTo(p0).withShape([N,1]))
    ucp.M.constraint(u.slice([0,0],[N,1]), Domain.equalsTo(u0).withShape([N,1]))

# Minimal downtime/uptime requirements
def consMinT(ucp, mdown, mup, u0, l0):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    # Different units will have different requirements
    for unit in range(N):
        # First, take care of neighbourhood of initial conditions
        if u0[unit] == 0 and l0[unit] < mdown[unit]:
            ucp.M.constraint(u.slice([unit,1],[unit+1,mdown[unit]-l0[unit]+1]), Domain.equalsTo(0))
        if u0[unit] == 1 and l0[unit] < mup[unit]:
            ucp.M.constraint(u.slice([unit,1],[unit+1,mup[unit]-l0[unit]+1]), Domain.equalsTo(1))
        # If we are off in i and i+w (w<mup), we must be off in between:
        for w in range(1,mup[unit]):
            for s in range(1,w):
                ucp.M.constraint(
                    Expr.sub(
                        Expr.add(u.slice([unit,0],[unit+1,T-w+1]), 
                                 u.slice([unit,w],[unit+1,T+1])),
                        u.slice([unit,s],[unit+1,T-w+1+s])),
                    Domain.greaterThan(0))
        # If we are on in i and i+w (w<mdown), we must be on in between:
        for w in range(1,mdown[unit]):
            for s in range(1,w):
                ucp.M.constraint(
                    Expr.sub(
                        Expr.add(u.slice([unit,0],[unit+1,T-w+1]), 
                                 u.slice([unit,w],[unit+1,T+1])),
                        u.slice([unit,s],[unit+1,T-w+1+s])),
                    Domain.lessThan(1))

# Random switch-off of some generators
def consSOff(ucp, num):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v   
    ucp.M.constraint(u.pick(list(np.random.randint(0,N,size=num)), list(np.random.randint(5,T,size=num))), 
        Domain.equalsTo(0.0))

# Objective
def objective(ucp, a, b, c, fx, sc, onlyLinear=False):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    # Startup costs and fixed operating costs
    scCost = Expr.dot(v.slice([0,1],[N,T+1]), repeat(sc, T))
    fxCost = Expr.dot(u.slice([0,1],[N,T+1]), repeat(fx, T))
    # Linear part of production cost bp
    bCost = Expr.sum(Expr.mul(b, p.slice([0,1],[N,T+1])))
    allCosts = [bCost, Expr.constTerm(sum(c)*ucp.T), fxCost, scCost]
    # Quadratic part of production cost ap^2
    if not onlyLinear:
        t = ucp.M.variable(ucp.N) 
        ucp.M.constraint(Expr.hstack(Expr.constTerm(N, 0.5), t, p.slice([0,1],[N,T+1])), Domain.inRotatedQCone())
        aCost = Expr.dot(a, t)
        allCosts.append(aCost)
    # Total
    ucp.M.objective(ObjectiveSense.Minimize, Expr.add(allCosts))

# Fetch results
def results(ucp):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v
    # Check if problem is feasible
    if ucp.M.getProblemStatus(SolutionType.Default) in [ProblemStatus.PrimalFeasible]:
        # For time-constrained optimization it may be wise to accept any feasible solution
        ucp.M.acceptedSolutionStatus(AccSolutionStatus.Feasible)
        
        # Some statistics:
        print('Solution status: ', ucp.M.getPrimalSolutionStatus())
        print('Relative optimiality gap: {:.2f}%'.format(100*ucp.M.getSolverDoubleInfo("mioObjRelGap")))
        print('Total solution time: {:.2f}s'.format(ucp.M.getSolverDoubleInfo("optimizerTime")))

        return p.level().reshape([N,T+1]), u.level().reshape([N,T+1])
    else:
        raise ValueError("No solution")

# Display some statistics
def displayProduction(ucp, pVal, uVal, pmax):
    N, T, p, u, v = ucp.N, ucp.T, ucp.p, ucp.u, ucp.v    
    f, axarr = plt.subplots(5, sharex=True)
    # Production relative to global maximum
    axarr[0].imshow(pVal/max(pmax), extent=(0,T+1,0,N),
               interpolation='nearest', cmap=cm.YlOrRd,
               vmin=0, vmax=1, aspect='auto')
    # Production relative to maximum of each unit
    axarr[1].imshow(pVal/repeat(pmax, T+1), extent=(0,T+1,0,N),
               interpolation='nearest', cmap=cm.YlOrRd,
               vmin=0, vmax=1, aspect='auto')
    # On/off status
    axarr[2].imshow(uVal, extent=(0,T+1,0,N),
               interpolation='nearest', cmap=cm.YlOrRd,
               vmin=0, vmax=1, aspect='auto')    
    # Number of units in operation
    axarr[3].plot(np.sum(uVal, axis=0))
    # Demand coverage and spinning reserve
    axarr[4].plot(demand, 'r', np.sum(repeat(pmax,T)*uVal[:,1:], axis=0), 'g')

    plt.show()

# Read data
example='RCUC/T-Ramp/20_0_3_w.mod'
numDays = 3

T, N, demand, pmin, pmax, rdown, rup, mdown, mup, a, b, c, sc, fx, p0, u0, l0 = ucpParser.loadModel(example, numDays)

ucp = UCP(T, N)
consProd(ucp, pmin, pmax, demand)
consRamp(ucp, rdown, rup)
consInit(ucp, p0, u0, l0)
consMinT(ucp, mdown, mup, u0, l0)
objective(ucp, a, b, c, fx, sc)

ucp.M.setLogHandler(sys.stdout)
ucp.M.setSolverParam("mioMaxTime", 30.0)
#ucp.M.setSolverParam("mioTolRelGap", 0.005)

#ucp.M.writeTask("ucp.task")

ucp.M.solve()

pVal, uVal = results(ucp)
displayProduction(ucp, pVal, uVal, pmax)

consSOff(ucp, 5)
ucp.M.solve()
pVal, uVal = results(ucp)
displayProduction(ucp, pVal, uVal, pmax)
