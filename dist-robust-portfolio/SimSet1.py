import sys
import time
import numpy as np
from scipy.special import erfinv
from distributionally_robust_portfolio import *


# Inherits from the portfolio class defined above.
class SimSet1(DistributionallyRobustPortfolio):
    # Market setup
    m = 10
    mu = np.arange(1, 11)*0.03
    var = 0.02 + (np.arange(1, 11)*0.025)

    # Constants for CVaR calculation.
    beta, rho = 0.8, 10
    c2_beta = 1/(np.sqrt(2*np.pi)*(np.exp(erfinv(2*beta - 1))**2)*(1-beta))

    def __init__(self, N, eps_range):
        self.eps_range = eps_range
        # Fusion model instance
        super().__init__(SimSet1.m, N)

    def run_sim(self, data_sets):
        '''
        Method to iterate over a list of independent datasets via the iter_data 
        generator method and save the results.
        '''
        wts, perf, rel = zip(*self.iter_data(data_sets))
        self.weights = np.mean(wts, axis=0)
        self.perf_mu = np.mean(perf, axis=0)
        self.perf_20 = np.quantile(perf, 0.2, axis=0)
        self.perf_80 = np.quantile(perf, 0.8, axis=0)
        self.reliability = np.mean(rel, axis=0)

    def simulate(self, data):
        '''
        Method called within the iter_data generator method, as defined in the 
        parent class.

        Returns
        wts: optimal asset weights for each radius
        perf: analytic out-of-sample performance for each radius
        rel: reliability for each radius
        '''
        # Set TrainData parameter
        self.dat.setValue(data)
        # Iterate through range of Wasserstein radii (see solve method below)
        wts, perf, rel = zip(*[(_w, _p, _r)
                               for _w, _p, _r in self.iter_radius(self.eps_range)])
        return wts, perf, rel

    def solve(self, epsilon):
        '''
        Method called within the iter_radius generator method, as defined in the
        parent class.

        Returns:
        x_sol: asset weights
        out_perf: analytica out-of-sample performance
        rel: reliability for the certificate
        '''
        # Set WasRadius parameter (TrainData is already set)
        self.eps.setValue(epsilon)
        # Solve the Fusion model
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        # Portfolio weights
        x_sol = self.x.level()
        # Analytical out-of-sample performance
        out_perf = self.analytic_out_perf(x_sol)
        return x_sol, out_perf, (out_perf <= self.M.primalObjValue())

    def analytic_out_perf(self, x_sol):
        '''
        Method to calculate the analytical value for the out-of-sample performance.
        [see Rockafellar and Uryasev]
        '''
        mean_loss = -np.dot(x_sol, SimSet1.mu)
        sd_loss = np.sqrt(np.dot(x_sol**2, SimSet1.var))
        cVaR = mean_loss + (sd_loss*SimSet1.c2_beta)
        return mean_loss + (SimSet1.rho*cVaR)
