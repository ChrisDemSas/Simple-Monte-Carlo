"""
This is an implementation of simple monte carlo, using the common distributions.

The following distributions are supported:

Univariate:

1) Normal
2) Uniform
3) Gamma
4) Beta
5) Exponential
"""

import numpy as np
import torch
import inspect
from UnivariateDistributions import *
import seaborn as sns
import pandas as pd


param = {'Mean': 0, # For Normal
         'Variance': 1, # For Normal
         'Alpha': 1, # For Gamma, Beta
         'Beta': 1, # For Gamma, Beta
         'Lambda': 3, # For Exponential
         'a': 1, # For Uniform
         'b': 2 # For Uniform
         }


class MonteCarlo:
    """Simple Monte Carlo Class.
    
    Parameters:
    
    n: Number of samples
    f: Function to Integrate
    """
    
    def __init__(self, function, upper: float, lower: float, sample_size: int):
        """Initialize the MonteCarlo Class.
        
        
        1) Note: does not support divisions yet.
        2) Preconditions:
        
        - Must be in the form of numpy stuff (np.exp, np.log etc.)
        - Any divisions must be in the form: a/b = a * b**-1
        
        """

        self.upper = upper
        self.lower = lower
        self.function = function
        self.sample_size = sample_size
        
        self.distribution = None
        self.distribution_name = None
        self.parameters = None
        
            
    def set_distribution(self, distribution: str, param: dict):
        """Set the distribution with the correct parameters."""
        
        self.distribution_name = distribution
        
        if distribution == 'Uniform':
            self.distribution = Uniform(param['a'], param['b'])
            self.parameters = {'a': param['a'], 'b': param['b']}
        
        elif distribution == 'Normal':
            self.distribution = Normal(param['Mean'], param['Variance'])
            self.parameters = {'Mean': param['Mean'], 'Variance': param['Variance']}
        
        elif distribution == 'Beta':
            self.distribution = Beta(param['Alpha'], param['Beta']) 
            self.parameters = {'Alpha': param['Alpha'], 'Beta': param['b']}
                            
        elif distribution == 'Gamma':
            self.distribution = Gamma(param['Alpha'], param['Beta'])
            self.parameters = {'Alpha': param['Alpha'], 'Beta': param['Beta']}
        
        elif distribution == 'Exponential':
            self.distribution = Exponential(param['Lambda'])
            self.parameters = {'Lambda': param['Lambda']}
    
    def check_distribution(self):
        """Returns the distribution with its parameters."""
        
        return f"Current Distribution: {self.distribution_name}, Parameter: {self.parameters}"
            
    def evaluate(self, sample_size = None):
        """Take in a distribution and evaluate the integral.
        """
        
        final = 0
        if sample_size is None:
            sample_size = self.sample_size
        
        for i in range(0, sample_size):
            sample = self.distribution.sample()
            
            if self.distribution_name == 'Uniform':
                final += self.function(sample, self.parameters['a'], self.parameters['b'])
            elif self.distribution_name == 'Exponential':
                final += self.function(sample, self.parameters['Lambda'])
            elif self.distribution_name == 'Beta':
                final += self.function(sample, self.parameters['Alpha'], self.parameters['Beta'])
            elif self.distribution_name == 'Gamma':
                final += self.function(sample, self.parameters['Alpha'], self.parameters['Beta'])
            elif self.distribution_name == 'Normal':
                final += self.function(sample, self.parameters['Mean'], self.parameters['Variance'])

        return final / sample_size
    
    def visualize_sample(self, lower_limit: int, upper_limit: int):
        """Visualize the results on the variation in samples, starting from the sample
        using a line graph."""
        
        samples = [n for n in range(lower_limit, upper_limit + 1)]
        dataset = []
        
        for i in range(lower_limit, upper_limit + 1):
            evaluation = self.evaluate(sample_size = i)
            dataset.append(evaluation)
        
        data = {'Samples': samples, 'Integral Value': dataset}
        dataframe = pd.DataFrame(data)
        sns.lineplot(data = dataframe, x = 'Samples', y = 'Integral Value')
        print(f'Mean {np.mean(dataset)} and Variance {np.var(dataset)}')
    
    def _float_range(self, lower_limit: float, upper_limit: float, step: float):
        """Return a list of values for the float."""
        
        final = [lower_limit]
        curr = lower_limit
        
        while curr < upper_limit:
            final.append(curr + step)
            curr += step
        return final
    
    def visualize_parameter(self, lower_limit: float, upper_limit: float, step: float):
        """Visualize the results on the variation of parameters."""
        
        parameter = self._float_range(lower_limit, upper_limit, step)
        data = []
        defaults = self.parameters

        if self.distribution_name == 'Uniform':
            a_data = []
            b_data = []
            for i in parameter:
                self.distribution.change_parameter(a = i)
                evaluation = self.evaluate()
                a_data.append(evaluation)
            self.parameters = defaults
            
            for i in parameter:
                self.distribution.change_parameter(b = i)
                evaluation = self.evaluate()
                b_data.append(evaluation)  
            self.parameters = defaults
            data = [('Alpha', a_data), ('Beta', b_data)]

        elif self.distribution_name == 'Exponential':
            for i in parameter:
                self.distribution.change_parameter(lamda = i)
                evaluation = self.evaluate()
                data.append(evaluation)
            data = [('Lambda', data)]
                
        elif self.distribution_name == 'Beta':
            alpha_data = []
            beta_data = []
            for i in parameter:
                self.distribution.change_parameter(alpha = i)
                evaluation = self.evaluate()
                alpha_data.append(evaluation)
            self.parameters = defaults
            for i in parameter:
                self.distribution.change_parameter(beta = i)
                evaluation = self.evaluation()
                beta_data.append(evaluation)
            self.parameters = defaults
            data = [('Alpha', alpha_data), ('Beta', beta_data)]
            
        elif self.distribution_name == 'Gamma':
            alpha_data = []
            beta_data = []
            for i in parameter:
                self.distribution.change_parameter(alpha = i)
                evaluation = self.evaluate()
                alpha_data.append(evaluation)
            self.parameters = defaults
            for i in parameter:
                self.distribution.change_parameter(beta = i)
                evaluation = self.evaluation()
                beta_data.append(evaluation)
            self.parameters = defaults
            data = [('Alpha', alpha_data), ('Beta', beta_data)]
            
        elif self.distribution_name == 'Normal':
            mean_data = []
            variance_data = []
            for i in parameter:
                self.distribution.change_parameter(mean = i)
                evaluation = self.evaluate()
                mean_data.append(evaluation)
            self.parameters = defaults
            for i in parameter:
                self.distribution.change_parameter(var = i)
                evaluation = self.evaluation()
                variance_data.append(evaluation)
            self.parameters = defaults
            data = [('Mean', mean_data), ('Variance', variance_data)]
        
        for d in data:
            dataframe = {'Parameter': parameter, 'Integral Value': d[1]}
            dataframe = pd.DataFrame(dataframe)
            sns.lineplot(data = dataframe, x = 'Parameter', y = 'Integral Value').set(title = d[0])
            print(f'Mean {np.mean(d[1])} and Variance {np.var(d[1])}')

if __name__ == '__main__':
    import random
    random.seed(0)
    
    function = lambda x, lamda: 1/3 * np.exp(lamda*x) * x**4 * np.cos(x)
    upper = np.inf
    lower = 0
    
    mc = MonteCarlo(function, upper, lower, 685)
    mc.set_distribution('Exponential', param)
    answer = mc.evaluate()
    #mc.visualize_sample(1, 690)
    mc.visualize_parameter(0.1, 0.2, 0.001)