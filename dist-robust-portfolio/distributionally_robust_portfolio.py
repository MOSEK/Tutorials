import sys
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from mosek.fusion import *
import mosek.fusion.pythonic


def normal_returns(m, N):
    R = np.vstack([np.random.normal(
        i*0.03, np.sqrt((0.02**2+(i*0.025)**2)), N) for i in range(1, m+1)])
    return (R.transpose())


class DistributionallyRobustPortfolio:

    def __init__(self, m, N):
        self.m, self.N = m, N
        self.M = self.portfolio_model(m, N)
        self.x = self.M.getVariable('Weights')
        self.t = self.M.getVariable('Tau')
        self.dat = self.M.getParameter('TrainData')
        self.eps = self.M.getParameter('WasRadius')
        self.sol_time = []

    def portfolio_model(self, m, N):
        '''
        Parameterized Fusion model for program in 9.
        '''
        M = Model('DistRobust_m{0}_N{1}'.format(m, N))
        ##### PARAMETERS #####
        dat = M.parameter('TrainData', [N, m])
        eps = M.parameter('WasRadius')
        a_k = [-1, -51]  # Alternative: Fusion parameters.
        b_k = [10, -40]  # Alternative: Fusion parameters.
        ##### VARIABLES #####
        x = M.variable('Weights', m, Domain.greaterThan(0.0))
        s = M.variable('s_i', N)
        l = M.variable('Lambda')
        t = M.variable('Tau')
        ##### OBJECTIVE #####
        # certificate = lamda*epsilon + sum(s)/N
        certificate = eps * l + Expr.sum(s) / N
        M.objective('J_N(e)', ObjectiveSense.Minimize, certificate)
        ##### CONSTRAINTS #####
        # b_k*t
        e1 = Expr.repeat(t * b_k, N, 1).T
        # a_k*<x,xi>
        e2 = Expr.hstack([a_k[i] * (dat @ x) for i in range(2)])
        # b_k*t + a_k*<x,xi> <= s
        M.constraint('C1', e1 + e2 <= Expr.repeat(s, 2, 1))
        # a_k*x
        e3 = Expr.hstack([a_k[i] * x for i in range(2)])
        e4 = Expr.repeat(Expr.repeat(l, m, 0), 2, 1)
        # ||a_k*x||_infty <= lambda
        M.constraint('C2_pos', e4 >= e3)
        M.constraint('C2_neg', e4 >= -e3)
        # x \in X
        M.constraint('C3', Expr.sum(x) == 1)
        # Use the simplex optimizer
        M.setSolverParam('optimizer', 'freeSimplex')
        return M


    def sample_average(self, x, t, data):
        '''
        Calculate the sample average approximation for given x and tau.
        '''
        l = np.matmul(data, x)
        return np.mean(np.maximum(-l + 10*t, -51*l - 40*t))

    def iter_data(self, data_sets):
        '''
        Generator method for iterating through values for the 
        TrainData parameter.
        '''
        for data in data_sets:
            yield self.simulate(data)

    def iter_radius(self, epsilon_range):
        '''
        Generator for iterating through values for the WasRadius
        parameter.
        '''
        for epsilon in epsilon_range:
            yield self.solve(epsilon)

    def simulate(self, data):
        '''
        Define in child classes.
        '''
        pass

    def solve(self, epsilon):
        '''
        Define in child classes.
        '''
        pass
