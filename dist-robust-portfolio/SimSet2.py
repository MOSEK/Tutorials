import sys
import time
import numpy as np
from scipy.special import erfinv
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import KFold, train_test_split
from distributionally_robust_portfolio import *

# Inherits from the portfolio class defined above.


class SimSet2_Holdout(DistributionallyRobustPortfolio):
    # Market setup
    m = 10

    # Radius range (see page 156 in Esfahani and Kuhn)
    eps_range = np.concatenate([np.arange(1, 10)*10.0**(i)
                                for i in range(-3, 0)])

    # Validation data set of 2*10**5 samples (page 158 in paper)
    valids = normal_returns(10, 2*10**5)

    def __init__(self, N, k=5):
        # 1/k sized split for the test data.
        self.k = k
        # Fusion model for (train) data size N*(k-1)/k (see holdout method)
        super().__init__(SimSet2_Holdout.m, np.rint(N*(k-1)/k).astype(np.int32))

    def validate(self, data_sets):
        '''
        Method to iterate over a list of independent datasets via the iter_data
        generator method so as to apply the holdout technique to each dataset 
        and then save the results.
        '''
        self.perf, self.cert, radii = zip(*self.iter_data(data_sets))
        self.rel = np.mean(np.array(self.perf) <= np.array(self.cert), axis=0)
        self.radius = np.mean(radii, axis=0)

    def simulate(self, data):
        '''
        Method called within the iter_data generator.

        Returns
        out_perf: out-of-sample performance calculated with validation data
        cert: performance certificate (optimal objective for M)
        eps_holdout: radius selected from holdout method
        '''
        # Split data into test and train
        train, self.test = train_test_split(data, test_size=1/self.k)
        # Set the TrainData parameter to train data
        self.dat.setValue(train)
        # Iterate through a range of Wasserstein radii
        out_perf_test, x, t, J_N = zip(
            *self.iter_radius(SimSet2_Holdout.eps_range))
        # Index of eps_holdout in the eps_range.
        min_arg = np.argmin(out_perf_test)
        # Out-of-sample performance for x_N(eps_holdout)
        out_perf = self.sample_average(
            x[min_arg], t[min_arg], SimSet2_Holdout.valids)
        # J_N(eps_holdout)
        cert = J_N[min_arg]
        return out_perf, cert, SimSet2_Holdout.eps_range[min_arg]

    def solve(self, epsilon):
        '''
        Method called within the iter_radius generator.

        Returns
        out_perf: SA-approx of out-of-sample performance using test data
        x: Portfolio weights
        t: Tau
        self.M.primalObjValue(): performance certificate
        '''
        # Set the WasRadius parameter
        self.eps.setValue(epsilon)
        # Solve the Fusion model
        self.M.solve()
        self.sol_time.append(self.M.getSolverDoubleInfo('optimizerTime'))
        # Weights and Tau optimal values
        x, t = self.x.level(), self.t.level()
        # SAA of out-of-sample performance based on test data
        out_perf = self.sample_average(x, t, self.test)
        return out_perf, x, t, self.M.primalObjValue()


# IMPORTANT: sub-class of the SimSet2_Holdout class!
class SimSet2_kFold(SimSet2_Holdout):

    def __init__(self, N, k=5):
        self.k = k
        # Object for holdout method (k-holdouts)
        super().__init__(N, k=k)
        # Fusion model for N-size dataset (results)
        self.M_N = self.portfolio_model(SimSet2_Holdout.m, N)
        self.dat_N = self.M_N.getParameter('TrainData')
        self.eps_N = self.M_N.getParameter('WasRadius')
        self.x_N = self.M_N.getVariable('Weights')
        self.t_N = self.M_N.getVariable('Tau')

    def simulate(self, data):
        '''
        Method called within the iter_data generator. This method overwrites
        the one defined in the SimSet2_Holdout class.

        Returns
        out_perf: out-of-sample performance calculated with validation data
        cert: performance certificate (optimal objective for M_N)
        eps_kFold: radius selected from k-Fold method
        '''
        # Set TrainData paremeter for M_N to data
        self.dat_N.setValue(data)
        # Perform the holdout method k times and calculate eps_kFold
        eps_kFold = np.mean([self._simulate(data) for i in range(self.k)])
        # Set WasRadius to mean from k holdout runs
        self.eps_N.setValue(eps_kFold)
        # Solve the M_N model.
        self.M_N.solve()
        # Out-of-sample performance for x_N(eps_kFold)
        out_perf = self.sample_average(
            self.x_N.level(), self.t_N.level(), SimSet2_Holdout.valids)
        # J_N(eps_kFold)
        cert = self.M_N.primalObjValue()
        return out_perf, cert, eps_kFold

    def _simulate(self, data):
        '''
        Method to perform the holdout technique for a given dataset. This 
        is called k times within each call to the simulate method. Works
        analogously to the simulate method of SimSet2_Holdout class.

        Returns:
        eps_holdout: WasRadius selected in one holdout run
        '''
        # Split data into test and train
        train, self.test = train_test_split(data, test_size=1/self.k)
        # Set TrainData parameter for the N*(k-1)/k model
        self.dat.setValue(train)
        # Solve N*(k-1)/k model iteratively for a range of radii
        saa, x, t, J_N = zip(*self.iter_radius(SimSet2_Holdout.eps_range))
        # Select Wasserstein radius that minimizes out-of-sample perf
        min_arg = np.argmin(saa)
        return SimSet2_Holdout.eps_range[min_arg]
