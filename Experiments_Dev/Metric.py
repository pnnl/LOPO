import torch
import numpy as np

import torch.nn as nn




class Identity(torch.nn.Module):
    # P_d is the diagonal
    def __init__(self,n_dim,parm_dim, P_d =  None, flat=False):
        super().__init__()

        self.n_dim = n_dim
        self.flat = flat
        #self.P_d = torch.nn.Parameter(torch.ones(self.n_dim)) if P_d==None else torch.diag(P_d)
        self.P_d = torch.nn.Parameter(torch.ones(self.n_dim)) if P_d==None else P_d

    def forward(self,x,parms):

        Pm = self.P_d if self.flat else torch.diag(self.P_d)

        return Pm

# A version of Identity (above), in which only the
#   vector of nonzero entries on the metric's diagonal
#   are returned.
class FlatIdentity(torch.nn.Module):
    # P_d is the diagonal
    def __init__(self,n_dim,parm_dim, P_d =  None):
        super().__init__()

        self.n_dim = n_dim
        self.P_d = torch.nn.Parameter(torch.ones(self.n_dim)) if P_d==None else P_d

    def forward(self,x,parms):

        Pm = self.P_d
        return Pm




class ParametricDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,flat=False,normalize_prod=False):
        super().__init__()

        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.flat = flat
        self.normalize_prod = normalize_prod



        self.parm_dim = parm_dim
        self.hidden_dim = self.parm_dim   # 10# self.parm_dim*2  self.parm_dim#

        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )



    def forward(self,x,parms):

        Pd = self.DiagMap(parms)

        P_diag = self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound)
        #if self.normalize_prod:
        #    P_diag = P_diag / P_diag.prod()

        Pm = P_diag if self.flat else torch.diag(P_diag)

        return Pm















"""
A version of Parametric Diagonal, where a single scaling factor is
predicted to scale a predefined diagonal matrix (default is the Identity)
"""
class ParametricIdentity(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,P_d=None,flat=False):
        super().__init__()

        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.flat = flat
        self.parm_dim = parm_dim
        self.hidden_dim = self.parm_dim   # 10# self.parm_dim*2  self.parm_dim#

        self.P_d = torch.nn.Parameter(torch.ones(self.n_dim)) if P_d==None else P_d

        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,1),
        )



    def forward(self,x,parms):

        scale_activation = self.DiagMap(parms)

        scale_factor = self.P_diag_lower_bound + torch.sigmoid(scale_activation)*(self.P_diag_upper_bound - self.P_diag_lower_bound)
        #P_diag = torch.clamp(Pd, self.P_diag_lower_bound, self.P_diag_upper_bound)
        Pm = self.P_d*scale_factor if self.flat else torch.diag(self.P_d)*scale_factor
        return Pm







class StaticDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim):
        super().__init__()

        self.n_dim = n_dim
        self.P_diag_upper_bound = 2.0
        self.P_diag_lower_bound = 0.5

        self.P_d = torch.nn.Parameter(torch.normal(0*torch.ones((self.n_dim)),.01*torch.ones((self.n_dim))))


    def forward(self,x,parms):

        P_diag = self.P_diag_lower_bound + torch.sigmoid(self.P_d)*(self.P_diag_upper_bound - self.P_diag_lower_bound)

        Pm = torch.diag(P_diag)

        return Pm







class ParametricStateDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound):
        super().__init__()

        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound



        self.input_dim = parm_dim + n_dim
        self.hidden_dim = 20

        self.DiagMap = nn.Sequential(
          nn.Linear(self.input_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )



    def forward(self,x,parms):

        xsp = torch.cat((x,parms),dim = -1)

        Pd = self.DiagMap(xsp)

        P_diag = self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound)

        Pm = torch.diag(P_diag)

        return Pm
