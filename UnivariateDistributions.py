"""
Implementation of library of univariate distributions:

1) Normal
2) Exponential
3) Uniform
4) Gamma
5) Beta

# Rewrite the sampling methods: x is the sample, not p(x)!
"""

import numpy as np
import torch 
from typing import Any
import scipy.integrate as integrate
from scipy.integrate import quad
import random

class DomainError(Exception):
    """Domain Error for error in domain."""
    
    def __init__(self, message = "Input not within domian range."):
        """Initialize the DomainError Exception."""
        
        self.message = message
        super().__init__(self.message)
    
    
class UnivariateDistribution:
    """Implementation of the class distributions, which contains certain methods
    for inheritence.
    
    """
    
    def __init__(self):
        """Initialize the distribution class."""
        
        super().__init__()
    
    def pdf(self, x: float):
        """Implementation of the pdf."""
        
        raise NotImplementedError
    
    def cdf(self, x: float):
        """Implementation of the cdf."""
        
        return quad(self.pdf, self.domain[0], x)[0]
    
    def expectation(self, function = lambda x: x):
        """Calculate the expectation given an input, where:
        
        E(f) = E(f(x)), where f is a function depending on x.
        
        Pre-Condition:
        f must depend on x.
        """
        
        curr = lambda x: self.pdf(x) * function(x)
        
        return quad(curr, self.domain[0], self.domain[1])[0]
    
    def variance(self, function = lambda x: x**2):
        """Calculate the variance given an input function."""
        
        return self.expectation(function) - (self.expectation)**2
    
    def sample(self):
        """Return a single sample from a distribution."""
        
        raise NotImplementedError
        

class Uniform(UnivariateDistribution):
    """Implementation of the Uniform Distribution.
    
    Parameters:
    
    a: lower bound
    b: upper bound
    
    """
    
    def __init__(self, a: float, b: float):
        """Initialize the Uniform Distribution."""
        
        if b >= a:
            self.a = a
            self.b = b
            self.domain = [self.a, self.b]
        else:
            raise DomainError(self)
        

    def pdf(self, x: float):
        """Compute the pdf at a point."""
        
        if x <= self.b and x >= self.a:
            return 1 / (self.b - self.a)
        else:
            raise DomainError(self)


    def cdf(self, x: float):
        """Calculate the cdf at a point."""
        
        if x <= self.b and x >= self.a:
            return x / (self.b - self.a)
        else:
            raise DomainError(self)
    
    def sample(self):
        """Return a single sample of the Uniform Distribution."""
        
        return random.uniform(self.a, self.b)
    
    def change_parameter(self, a = None, b = None):
        """Change the parameter of the Uniform Distribution."""
        
        if a is not None:
            self.a = a
        if b is not None:
            self.b = b


class Exponential(UnivariateDistribution):
    """Implementation of the Exponential Distribution.
    
    
    Parameters:
    lamda: float (wrong spelling on purpose)
    """
    
    def __init__(self, lamda: float):
        """Initialize the Exponential Distribution."""
        
        self.lamda = lamda
        self.domain = [0, np.inf]
    
    def pdf(self, x: float):
        """Compute the pdf at a point."""
        
        if x >= 0:
            return self.lamda * np.exp(- self.lamda * x)
        else:
            raise DomainError(self)
    
    def cdf(self, x: float):
        """Compute the cdf at a point."""
        
        if x >= 0:
            return 1 - np.exp(- self.lamda * x)
        else:
            raise DomainError(self)
    
    def sample(self):
        """Return a sample from an Exponential Distribution."""
        
        rand = random.uniform(0, 1)
        
        return - np.log(1 - rand) / self.lamda
    
    def change_parameter(self, lamda = None):
        """Change the parameter on the Exponential Distribution."""
        
        if lamda is not None:
            self.lamda = lamda
    
    
class Normal(UnivariateDistribution):
    """Implementation of the Normal Distribution.
    
    Parameters:
    
    mean: float
    variance: float
    """
    
    def __init__(self, mean: float, variance: float):
        """Initialization of the Normal Distribution."""
        
        self.mean = mean
        self.variance = variance
        self.domain = [-np.inf, np.inf]
    
    def pdf(self, x: float):
        """Evaluate the pdf at a point."""
        
        return (1 / (self.variance * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - self.mean)/self.variance) **2)
    
    def sample(self):
        """Return a sample from an Normal Distribution.
        
        Method Used: Box Muller
        """
        
        u_1 = random.uniform(0, 1)
        u_2 = random.uniform(0, 1)
        
        z_1 = np.sqrt(-2 * np.log(u_1)) * np.cos(2 * np.pi * u_2)
        
        return self.variance * z_1 + self.mean
    
    def change_parameter(self, mean = None, var = None):
        """Change the parameter on the Normal Distribution."""
        
        if mean is not None:
            self.mean = mean
        if var is not None:
            self.variance = var
        

class Gamma(UnivariateDistribution):
    """Implementation of the Gamma Distribution.
    
    Parameters:
    
    alpha: float
    beta: float
    
    """
    
    def __init__(self, alpha: float, beta: float):
        """Initialize the Gamma distribution."""
        
        if alpha > 0 and beta > 0:
            self.alpha = alpha
            self.beta = beta
            self.domain = [0, np.inf]
        else:
            raise DomainError(self)
    
    def _gamma(self, z: float):
        """Implementation of the gamma function."""
        
        fxn = lambda t: t ** (z - 1) * np.exp(-t)
        
        return quad(fxn, 0, np.inf)[0]
    
    def pdf(self, x: float):
        """Calculate the pdf at a point x."""
        
        const = (self.beta ** self.alpha) / self._gamma(self.alpha)
        
        if x < 0:
            raise DomainError(self)
        else:
            return const * (x ** (self.alpha - 1)) * np.exp(- self.beta * x)
    
    def sample(self):
        """Return a sample from the Gamma distribution.
        
        Method: Marsling-Tsang.
        Small alpha
        """

        if self.alpha < 1:
            l = (1/self.alpha) - 1
            w = self.alpha / (np.exp(1) * (1 - self.alpha))
            u = random.uniform(0, 1)
            r = 1 / (1 + w)
            c = self._gamma(self.alpha + 1)
            
            if u <= r:
                z = - np.log(u/r)
            else:
                z = np.log(random.uniform(0, 1)) / l
            
            h = np.exp(-z-np.exp(-z/self.alpha)) / c
            
            if z < 0:
                n = c * w * l * np.exp(l * z)
            else:
                n = c * np.exp(-z)
            
            while h/n <= random.uniform(0, 1):
                u = np.random.uniform(0, 1)
                
                if u <= r:
                    z = - np.log(u/r)
                else:
                    z = np.log(random.uniform(0, 1)) / l
                
                h = np.exp(-z-np.exp(-z/self.alpha)) / c
                
                if z < 0:
                    n = c * w * l * np.exp(l * z)
                else:
                    n = c * np.exp(-z)  
                
            else:
                return np.exp(-z/self.alpha) / self.beta
        else:
            d = self.alpha - (1/3)
            c = 1/np.sqrt(9*d)
            
            normal = Normal(0, 1)
            x = normal.sample()
            v = (1+c*x)**3
            
            u = random.uniform(0, 1)
            
            while u >= 1 - 0.0331*x**4 or np.log(u) >= 0.5*x**2 + d*(1-v + np.log(v)):
                d = self.alpha - (1/3)
                c = 1/np.sqrt(9*d)  
                x = normal.sample()
                v = (1+c*x)**3            
                u = random.uniform(0, 1)
            else:
                return d*v/self.beta
    
    def change_parameter(self, alpha = None, beta = None):
        """Change the parameter on the Gamma Distribution."""
        
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        
            
class Beta(UnivariateDistribution):
    """Implementation of the Beta distribution class.
    
    Parameters:
    
    alpha: float
    beta: float
    """
    
    def __init__(self, alpha: float, beta: float):
        """Initialize the Beta Distribution."""
        
        if alpha > 0 and beta > 0:
            self.beta = beta
            self.alpha = alpha
            self.domain = [0, 1]
        else:
            raise DomainError(self)
    
    def _gamma(self, z: float):
        """Implementation of the gamma function."""
        
        fxn = lambda t: t ** (z - 1) * np.exp(-t)
        
        return quad(fxn, 0, np.inf)[0]
    
    
    def pdf(self, x: float):
        """Compute the pdf at a point."""
        
        const = self._gamma(self.alpha) * self._gamma(self.beta)/(self._gamma(self.beta + self.alpha))
        
        if x < 1 and x > 0: 
            return const * x ** (self.alpha - 1) * (1 - x) ** (self.beta - 1)
        else:
            raise DomainError(self)
    
    def sample(self):
        """Return a sample from the Beta distribution."""
        
        a_gamma = Gamma(self.alpha, 1)
        b_gamma = Gamma(self.beta, 1)
        
        a_sample = a_gamma.sample()
        b_sample = b_gamma.sample()
        
        return a_sample / (a_sample + b_sample)
    
    def change_parameter(self, alpha = None, beta = None):
        """Change the parameters on the Beta Distribution."""
        
        if alpha is not None:
            self.alpha = alpha
        if beta is not None:
            self.beta = beta
        

if __name__ == '__main__':
    import seaborn as sns
    
    x_data = []
    y_data = []
    n = 100
    distribution = Gamma(0.5, 1)
    
    for i in range(0, n):
        sample = distribution.sample()
        pdf = distribution.pdf(sample)
        x_data.append(sample)
        y_data.append(pdf)
    
    sns.lineplot(x_data, y_data)