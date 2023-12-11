import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
'''
Sample extremogram and cross extremogram with credible confidence bands, and a random
permutation scheme to test whether the extremal correlation is significant. 

The extremogram is given as 
    \lim_{x to \infty} P(X_{t+h} \in A| X_{t} \in B)
where (X)_{t \in N} is a stationary and regularaly varying sequence, and  A and B are sets
bounded away from 1. Regularaly varying sequence means, more or less, that the noise term in the 
time series is heavy tailed, for example Pareto distribued or student t-distributed.    
The extremogram examines the correlation between an extremal event at time t and time t+h. 
In certain cases, a closed form solution of the extremogram exists, for example for an AR(1) process. 

When we run the sample version of this metic, we need to choose how to define extremes. 
We do this using higher and lower quantiles of the data. However, there is a tradeoff. 
If we choose the quantile too high/low, the sample extremogram collapes, as it does not have enough
data. If, however, we choose is too low/high, we are not really considering extremes. 
As a rule of thump, we use the 95, 97.5 or 99 percent quantile, depending of the amount of data 
available. 
In this script, the default setting is 95% quantile. 

The class extremogram_class takes two pandas dataframes as input, and have the following methods: 

Uniform extremogram: 
    returns the uniform extremogram of the first dataset that was passed to the constructor. 

Plot_uniform_extrmogram: 
    plots the sample extremogram with confidence bands and compares the data to the independent case, 
    thus shows if the extremal correlation is significant. 

The rest of the methods are used internally in the class. You are more than welcome to edit, modify
or rewrite/optimize all of it.
'''


class extremogram_class:
    def __init__(self, data1, data2 = None):
        '''

        :param data1: Pandas dataframe. Used for the method extremogram_1. Must be in usual OPHLA form.
        :param data2: Pandas dataframe. Used for the method crossextremogram, which calculates the
        cross extreme effect between two time series.
        '''
        self.data_1 = data1
        self.returns_1 = data1.pct_change().dropna()
        if data2 is not None:
            self.data_2 = data2
            self.returns_2 = data2.pct_change().dropna()

    def show_returns(self, type = 'Close'):
        plt.plot(self.returns_1[type])
        plt.show()

    def show_dens(self):
        bin = np.histogram(self.returns_1.Close, bins = 100)[1]
        plt.hist(self.returns_1.Close, bins=bin, density=True)
        plt.show()

    def uniform_extremogram(self, data, quant_1, quant_2, lag = 30, type = 1):

        if type == 1:
            level_1 = np.quantile(data, quant_1)
            level_2 = np.quantile(data, quant_2)
        if type == 2:
            level_1 = np.quantile(data, 1-quant_1)
            level_2 = np.quantile(data, 1-quant_2)
        if type == 3:
            level_1 = np.quantile(data, quant_1)
            level_2 = np.quantile(data, 1 - quant_2)
        if type == 4:
            level_1 = np.quantile(data, 1 - quant_1)
            level_2 = np.quantile(data, quant_2)

        res = np.zeros(lag)
        n = len(data)
        if type == 1:
            for i in range(lag):
                cond1 = data[:(n - i)] > level_1
                cond2 = data[i:n] > level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]
        if type == 2:
            for i in range(lag):
                cond1 = data[:(n - i)] < level_1
                cond2 = data[i:n] < level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]
        if type == 3:
            for i in range(lag):
                cond1 = data[:(n - i)] > level_1
                cond2 = data[i:n] < level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]
        if type == 4:
            for i in range(lag):
                cond1 = data[:(n - i)] < level_1
                cond2 = data[i:n] > level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]
        res[0] = 1
        return res

    def cross_extremogram(self, dataset_1, dataset_2, quant_1, quant_2, lag=30, type=1):
        if type == 1:
            level_1 = np.quantile(dataset_1, quant_1)
            level_2 = np.quantile(dataset_2, quant_2)
        if type == 2:
            level_1 = np.quantile(dataset_1, 1-quant_1)
            level_2 = np.quantile(dataset_2, 1-quant_2)
        if type == 3:
            level_1 = np.quantile(dataset_1, quant_1)
            level_2 = np.quantile(dataset_2, 1 - quant_2)
        if type == 4:
            level_1 = np.quantile(dataset_1, 1 - quant_1)
            level_2 = np.quantile(dataset_2, quant_2)

        n = len(dataset_1)
        res = np.zeros(lag)
        if type == 1:
            for i in range(lag):
                cond1 = dataset_1[:(n - i)] > level_1
                cond2 = dataset_2[i:n] > level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]

        if type == 2:
            for i in range(lag):
                cond1 = dataset_1[:(n - i)] < level_1
                cond2 = dataset_2[i:n] < level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]

        if type == 3:
            for i in range(lag):
                cond1 = dataset_1[:(n - i)] > level_1
                cond2 = dataset_2[i:n] < level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]

        if type == 4:
            for i in range(lag):
                cond1 = dataset_1[:(n - i)] < level_1
                cond2 = dataset_2[i:n] > level_2
                res[i] = np.shape(np.where(cond1 & cond2))[1]
                res[i] = res[i] / np.shape(np.where(cond1))[1]

        return res

    def plot_uniform_extremogram(self, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1):
        ext = self.uniform_extremogram(
            data = np.array(self.returns_1['Close']),
            quant_1=quant_1,
            quant_2=quant_2,
            lag = lag,
            type = type
        )

        conf = self.generate_conf(
            dataset = np.array(self.returns_1['Close']),
            n_sims = 1000,
            mean_block_size=100,
            quant_1 = quant_1,
            quant_2 = quant_2,
            type = type,
            lag = lag
        )

        perm = self.is_significant(
            dataset=np.array(self.returns_1['Close']),
            n_sims=1000,
            quant_1=quant_1,
            quant_2=quant_2,
            type=type,
            lag=lag
        )

        plt.plot(ext, 'r', label = 'Sample extremogram')
        plt.plot(np.linspace(1, lag-1, lag-1), perm.T[1: ], 'b:', alpha = 0.2, label = 'Random Permutations')
        plt.plot(np.linspace(1, lag - 1, lag - 1), conf.T[1:], 'g--', alpha=0.2, label = 'Stationary Bootstrap method')
        plt.grid()
        plt.legend(loc = 'upper right')
        plt.show()

    def plot_cross_extremogram(self, dataset_1 = None, dataset_2=None, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1, mean_block_size = 100, n_sims = 1000):
        '''

        :param dataset_1: a numpy array of a stationary time-series.
        :param dataset_2: a numpy array of a stationary time-series.
        :param quant_1: float between (0, 1). quantile for the set A
        :param quant_2: float between (0, 1). quantile for the set B
        :param lag: int. amount of time to be considered.
        :param type: Int between (1, 4), If 1, upper/upper, 2, lower/lower, 3, upper/lower, 4, lower/upper
        :param mean_block_size: float. used for the stationary bootstrap. Decides the mean size of the samples
        :param n_sims: int. Amount of permutations for the creation of teh confidence bands.
        :return: plot with extremogram with confidence bands + hypothesis the extremes are independent.
        '''
        if dataset_1 is None:
            dataset_1 = np.array(self.returns_1['Close'])
            dataset_2 = np.array(self.returns_2['Close'])

        ext = self.cross_extremogram(
            dataset_1=dataset_1,
            dataset_2 = dataset_2,
            quant_1=quant_1,
            quant_2=quant_2,
            lag=lag,
            type=type
        )
        conf = self.generate_conf_cross(
            dataset_1=dataset_1,
            dataset_2=dataset_2,
            n_sims=n_sims,
            mean_block_size=mean_block_size,
            quant_1=quant_1,
            quant_2=quant_2,
            type=type,
            lag=lag
        )

        perm = self.is_significant_2(
            dataset_1=dataset_1,
            dataset_2 = dataset_2,
            n_sims=n_sims,
            quant_1=quant_1,
            quant_2=quant_2,
            type=type,
            lag=lag
        )

        plt.plot(ext, 'r', label='Sample extremogram')
        plt.plot(np.linspace(1, lag - 1, lag - 1), perm.T[1:], 'b:', alpha=0.5, label='Random Permutations')
        plt.plot(np.linspace(1, lag - 1, lag - 1), conf.T[1:], 'g--', alpha=0.2, label='Stationary Bootstrap method')
        plt.grid()
        plt.legend(loc='upper right')
        plt.show()


    def generate_stationary_bootsstap_method(self, dataset, n_sims, mean_block_size):
        n = len(dataset)
        res = np.zeros(shape = (n_sims, n))
        mean_block_size = 1/mean_block_size

        for i in range(n_sims):
            temp = np.array([])
            while np.shape(temp)[0] < n:
                start = np.random.randint(0, n, 1)[0]
                length = np.random.geometric(mean_block_size, 1)[0]
                temp = np.append(temp, dataset[start:(start+length)])
            res[i] = temp[:n]
        return res

    def random_permutations(self, dataset, n_sims):
        n = len(dataset)
        res = np.zeros(shape = (n_sims, n))
        for i in range(n_sims):
            res[i] = np.random.choice(dataset, size = n)
        return res

    def generate_conf(self, dataset, n_sims, mean_block_size, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1, alpha = 0.05):
        resample = self.generate_stationary_bootsstap_method(
            dataset=dataset,
            n_sims=n_sims,
            mean_block_size=mean_block_size
        )

        temp = np.zeros(shape = (n_sims, lag))

        for i in range(n_sims):
            temp[i] = self.uniform_extremogram(
                resample[i],
                quant_1=quant_1,
                quant_2=quant_2,
                lag = lag,
                type = type
            )

        confs = np.zeros(shape = (2, lag))
        for i in range(1, lag):
            confs[0, i] = np.quantile(temp[:, i], q = alpha/2)
            confs[1, i] = np.quantile(temp[:, i], q = 1-alpha/2)

        return confs

    def generate_conf_cross(self, dataset_1, dataset_2, n_sims, mean_block_size, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1, alpha = 0.05):
        resample_1 = self.generate_stationary_bootsstap_method(
            dataset=dataset_1,
            n_sims=n_sims,
            mean_block_size=mean_block_size
        )
        resample_2 = self.generate_stationary_bootsstap_method(
            dataset=dataset_2,
            n_sims=n_sims,
            mean_block_size=mean_block_size
        )

        temp = np.zeros(shape=(n_sims, lag))

        for i in range(n_sims):
            temp[i] = self.cross_extremogram(
                resample_1[i],
                resample_2[i],
                quant_1=quant_1,
                quant_2=quant_2,
                lag = lag,
                type = type
            )

        confs = np.zeros(shape = (2, lag))
        for i in range(1, lag):
            confs[0, i] = np.quantile(temp[:, i], q = alpha/2)
            confs[1, i] = np.quantile(temp[:, i], q = 1-alpha/2)

        return confs

    def is_significant(self, dataset, n_sims, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1, alpha = 0.05):
        resample = self.random_permutations(
            dataset=dataset,
            n_sims=n_sims
        )

        temp = np.zeros(shape=(n_sims, lag))

        for i in range(n_sims):
            temp[i] = self.uniform_extremogram(
                resample[i],
                quant_1=quant_1,
                quant_2=quant_2,
                lag=lag,
                type=type
            )

        perms = np.zeros(shape=(2, lag))
        for i in range(1, lag):
            perms[0, i] = np.quantile(temp[:, i], q=alpha / 2)
            perms[1, i] = np.quantile(temp[:, i], q=1 - alpha / 2)

        return perms

    def is_significant_2(self, dataset_1, dataset_2, n_sims, quant_1 = 0.95, quant_2 = 0.95, lag = 30, type = 1, alpha = 0.05):
        resample_1 = self.random_permutations(
            dataset=dataset_1,
            n_sims=n_sims
        )
        resample_2 = self.random_permutations(
            dataset = dataset_2,
            n_sims=n_sims
        )

        temp = np.zeros(shape=(n_sims, lag))

        for i in range(n_sims):
            temp[i] = self.cross_extremogram(
                resample_1[i],
                resample_2[i],
                quant_1=quant_1,
                quant_2=quant_2,
                lag=lag,
                type=type
            )

        perms = np.zeros(shape=(2, lag))
        for i in range(1, lag):
            perms[0, i] = np.quantile(temp[:, i], q=alpha / 2)
            perms[1, i] = np.quantile(temp[:, i], q=1 - alpha / 2)

        return perms


'''
Example of usage. Here we use yahoo finance to download the appel stock and the SP500 index. 
The uniform extremogram and the cross-extremogram is calculated between the two datasets. 
'''

data_1 = yf.download('AAPL',
                   start = '2010-01-01',
                   end = '2021-01-01')

data_2 = yf.download('SPY',
                   start = '2010-01-01',
                   end = '2021-01-01')
# parameters used
quant_1 = 0.9
quant_2 = 0.9
lag = 30
type = 1

e1 = extremogram_class(data_1, data_2) # Initiate the class

e1.plot_uniform_extremogram(quant_1 = quant_1, quant_2 = quant_2, type = type, lag = lag) # Plot the uniform extremogram.

e1.plot_cross_extremogram(quant_1 = quant_1, quant_2 = quant_2, type = type, lag = lag)

# These examples show little, to no significant extremal correlation.
# Another example, where there is significant extremal correlation.
# Here we consider the extremal correlation between the 'open' and 'close' price of the same stock in absolute returns.
# Uncomment to run the example.
'''
e1.plot_cross_extremogram(
    dataset_1 = np.abs(np.array(e1.returns_1['Close'])),
    dataset_2 = np.abs(np.array(e1.returns_1['Open'])),
    quant_1 = quant_1,
    quant_2 = quant_2,
    type = type,
    lag = lag
)'''