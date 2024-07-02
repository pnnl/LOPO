# Learning to Optimize With Proximal Operators 
This repo contains implementations of two operator splitting solvers Douglas Rachford (DR) and Alternating Direction Method of Multipliers (ADMM) for solution of general optimization problems by second order approximations. In addition it contains a parameteric implementation to learn metrics for proximal operators to speed up convergence in the parameteric setting. This is the work presented in this paper https://arxiv.org/abs/2404.00882 . Note though that in the implementation here metric learning is performed in an unsupervised fashion for ease of use.


# Contents
This Repo contains several examples:

1. pQp example: This is an example highlighting the effect of metric learning on a toy quadratic programming problem
2. porfolio example: This example demonstrates performance on a larger quadratic problem
3. Quadcopter example: Shows application of the methodology for a model predictive control application

4. LOPO: Contains the relevant code that implements the solver methods.
