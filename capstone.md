# Capstone Report Proposal #

## Hyperparameter Selection for LightGBM using Bayesian Optimization ##

The tuning of machine learning models (design or hyperparameter selection) can be a
difficult process for many machine learning models.  One such example would be
the computationally intensive deep learning networks.  Default hyperparameter setups that are found for
example in Scikit-learn can be reasonably good for many datasets but often there are substantial gains to
be captured from tuning the hyperparameters. Such tuning is typically scored by the minimization of k-fold
cross-validation error on the training set.  The simplest techniques for tuning hyperparameters is grid search
and random search.  Grid search is an exhaustive search through a discrete selection or grid of each hyperparameter.  
This method will practically fail as the dimension of the hyperparameter vector increases.  This is known as
the curse of dimensionality.  Random search while easy to parallelize doesn't learn from the results of
previous experiments.  Bayesian optimization attempts to find an optimum of an unknown function using
a minimum of expensive function calls.  In our case of hyperparameter tuning, the unknown function is
the cross-validation score of model over the space defined by the vector of hyperparameters.  Bayesian
optimization is a general optimization scheme and can be used for many other applications such as
industrial design, chemical research and geostatistics.

The main idea of bayesian optimization is to compute a posterior over the unknown function given the few known
function evaluations and then use the posterior distribution to select good points to evaluate next.  We use affects
surrogate function which is easy to compute in order to approximate the expensive true cost function (hyperparameter score).
The surrogate function most commonly used to define the posterior distribution is the gaussian process.  A
gaussian process is a collection of space and/or time indexed random variables that any subset of have an
multivariate Gaussian distribution. The primary choice that affects the properties of a gaussian process is that
of the covariance (kernel) function.  This is a function that specifies the covariance of any two points in the
gaussian process.  Examples of kernel functions are the squared exponential, Ornstein-Uhlenbeck,
Matern, Periodic, Brownian, and many others.  Each of these kernel functions can define a surrogate
with dramatically different properties.  Another crucial issue is where to search next given our current models.
We have a tradeoff between exploiting our current information and reducing uncertainty in the future by exploring
new areas.  In bayesian optimization, many of these acquisition strategies exist such as expected improvement, probability of
improvement, and a generalized expected improvement.

I propose to experiment with data science methods such as ensembling to improve the performance of bayesian
optimization for hyperparameter tuning.  Specifically, I will run in parallel an ensemble of gaussian processes
to reduce estimation variance.  The literature commonly points out the differing properties of the acquisition
functions used in practice.  I would also experiment in using multiple acquisition functions as a way of balancing
their respective strengths and weaknesses.  This is a relatively simple idea that amazingly seems to have been
overlooked in the research papers that I have found so far.   Initial experiments will be conducted on a variety
of low dimensional test functions that have been used prevously in comparisons.  The ultimate test will be on the current Kaggle
contest dataset "Home Credit Default Risk" using LightGBM as the predictive model.  LightGBM is a popular implementation of gradient
boosting for decision trees and has many hyperparameters available for tuning.  My initial experiments will be conducted
with the libraries GPyOpt and BayesianOptimization.  The biggest difficulties will be involved in
the parallel programming and message passing of processes for the separate surrogate models.  I would evenually (quite down the road) want to
expand the search ideas using direct search methods and implement the library on a GPU. Different methods for weighting and creating new
acquisition functions can be explored eventually.  My ultimate plan would be to release my own library for hyperparameter
tuning that would be GPU ready with the heavy computation done in Cython or C++.