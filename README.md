# Douglas Rachford Solvers
This contains an implementation of Douglas Rachford for solving general optimization problems by first and second order approximations. In addition it contains a parameteric implementation of DR that learns metrics for proximal operators to speed up DR convergence in the parameteric setting 


# Contents
This Repo contains several implementations:

1. DRcorrection: This is a nonparameteric implementation of the Douglas Rachford algorithm
2. ParaMetricDR: This containtains a metric learning implementation of Douglas Rachford for parametric problems.

## DR correction
This folder contains:
1. DRSolver.py: The implementation of Douglas Rachford for solution correction
2. pQP_Example.py: An example implementation on a parameteric quadratic programming problem
3. RosenBrock_Example.py: An example implementation on a nonlinear, nonconvex programming problem


## ParaMetricDR
This folder contains:
1. ParaMetricDRSolver.py: The implementation of a metric learning Douglas Rachford for solution correction
2. pQP_Example.py: An example implementation on a parameteric quadratic programming problem
3. RosenBrock_Example.py: An example implementation on a nonlinear, nonconvex programming problem




