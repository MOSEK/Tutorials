import numpy as np
import matplotlib.pyplot as plt
from distributionally_robust_portfolio import *
from SimSet2 import *


class SimSet3(DistributionallyRobustPortfolio):
    m = 10
    eps_range = np.concatenate([np.arange(1, 10)*10.0**(i)
                                for i in range(-3, 0)])
    valids = normal_returns(10, 2*10**5)

    def __init__(self, beta, N, k=50):
        # Number of resamples
        self.k = k
        # Reliability threshold
        self.beta = beta
        # Instantiate Fusion model
        super().__init__(SimSet3.m, N)

    def bootstrap(self, data_sets):
        '''
        Method to iterate over a list of independent datasets via the iter_data
        generator method so as to apply the bootstrap technique to each dataset
        and then save the results.
        '''
        self.perf, self.cert, radii = zip(*self.iter_data(data_sets))
        self.rel = np.mean(np.array(self.perf) <= np.array(self.cert), axis=0)
        self.radii = np.mean(radii, axis=0)

    def simulate(self, data):
        '''
        Method called within the iter_data generator.

        Returns
        out_perf: out-of-sample performance calculated with validation data
        cert: performance certificate (optimal objective for M)
        eps_btstrp: radius selected from holdout method
        '''
        # List to store reliability
        rel = []
        # Perform k resamples
        for i in range(self.k):
            # Split data into test and train
            train, self.test = train_test_split(data, test_size=1/3)
            # Resample train data up-to size N
            train = resample(train, n_samples=self.N)
            # Set TrainData parameter to train
            self.dat.setValue(train)
            # Iterate through a range of Wasserstein radii
            rel.append(self.iter_radius(SimSet3.eps_range))
        # Sum reliability over all resamples (for each epsilon)
        rel = np.sum(rel, axis=0)
        # Smallest radius that has reliability over 1-beta
        _id = next(i for i, r in enumerate(rel) if r >= self.k*(1-self.beta))
        eps_btstrp = SimSet3.eps_range[_id]
        # Set TrainData parameter to data
        self.dat.setValue(data)
        # Set WasRadius parameter to eps_btstrp
        self.eps.setValue(eps_btstrp)
        self.M.solve()
        # Out-of-sample performance for x_N(eps_btstrp)
        out_perf = self.sample_average(
            self.x.level(), self.t.level(), SimSet3.valids)
        cert = self.M.primalObjValue()
        return out_perf, cert, eps_btstrp

    def solve(self, epsilon):
        '''
        Method called within the iter_radius generator.

        Returns
        reliability: SAA of out-of-sample performance <= certificate(epsilon)
        '''
        # Set WasRadius parameter to epsilon and solve
        self.eps.setValue(epsilon)
        self.M.solve()
        # Calculate out-of-sample performance SAA estimator using test
        saa = self.sample_average(self.x.level(), self.t.level(), self.test)
        # Boolean to state if the certificate is greater than SAA estimate
        return saa <= self.M.primalObjValue()
