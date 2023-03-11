# Simple-Monte-Carlo
This is an implementation of the Simple Monte Carlo Algorithm. Integration is typically very difficult for non-elementary functions because of a lack of closed form and so mathematicians need a numerical way to integrate a non-elementary function. One of the ways to do this is through the Simple Monte Carlo method, which has the following steps:

1) Pick a probability distribution
2) Generate samples from the probability distribution 
3) Use the Law of Large Numbers in order to calculate the sample expectation and evaluate the integral

The choice of function for the expected value depends on the function. In this code, we want to integrate the function:

$$\int^{\infty}_{0} x^{4}cos(x) dx$$

In order to do this, we choose to sample from the exponential distribution because of the domain, $x \in (0, \infty)$. We therefore manipulate the above integral to become:

$$\int^{\infty}_{0} x^{4}cos(x) dx = \int^{\infty}_{0} x^{4}cos(x) \frac{\lambda}{\lambda} e^{(\lambda - \lambda)x} = E(\frac{1}{\lambda}e^{\lambda x} x^{4})dx$$ 

We note that there is, in fact, an analytical solution to the integral:
$$\int x^{4}cos(x) dx = (x^{4} - 12x^{2} + 24)sin(x) + 4xcos(x)(x^{2} - 6) + C$$

Which implies that the definite integral diverges. Indeed, when we look at the values after sampling, we obtain the following graph, using samples between $n = 1$ to $n = 1000$:
<img width="846" alt="Screenshot 2023-03-11 at 1 13 56 PM" src="https://user-images.githubusercontent.com/93426725/224504949-c63429e1-9f6f-48f4-abd6-0b71feb99d13.png">

Figure 1: Figure showing the fluctuations in the value of the integral using different sample sizes. Since the values of the integral fluctuates as the sample increases (no convergence), we can see that the integral diverges.

It is interesting to note, that the value obtained by the integral, when we vary the parameters, converges towards 0. In fact, this solution is agreed upon by the integral calculator: https://www.integral-calculator.com/. We get the following graph, using the sample $n = 1000$:

<img width="628" alt="Screenshot 2023-03-11 at 1 23 07 PM" src="https://user-images.githubusercontent.com/93426725/224505262-909faadc-6968-4203-bbf4-e3e9002ac9af.png">

Figure 2: Figure showing the variation of parameter and its' effect on the value of the integral.

In fact, that this does not agree with the sample size diagnostic is interesting. However, when we consider the fact that the probability distribution shape is very sensitive to $\lambda$, we see that the higher the $\lambda$ value, the more L shaped the graph becomes. Therefore, we can conclude that the higher the parameter, the more likely samples are coming from a smaller subset between 0 and infinity. Due to this sensitivity, we see that even in smaller $\lambda$ values, the values are still quite large, thus showing the divergence of the integral:

<img width="614" alt="Screenshot 2023-03-11 at 1 40 18 PM" src="https://user-images.githubusercontent.com/93426725/224505951-d2e01795-4d23-43c5-8b43-3966028d6830.png">

Figure 3: Figure showing the high values of parameters between $\lambda \in [1, 2]$.






