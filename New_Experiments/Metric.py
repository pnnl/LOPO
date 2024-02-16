import torch
import numpy as np

import torch.nn as nn



 
class Identity(torch.nn.Module):
    def __init__(self,n_dim,parm_dim):
        super().__init__()
        self.n_dim = n_dim
        self.P_d = torch.nn.Parameter(torch.ones(self.n_dim),requires_grad = False)
        self.scl = torch.nn.Parameter(torch.ones(1),requires_grad = False)
    def forward(self,x,parms):
        Pm = torch.diag(self.P_d)
        return Pm
    def scl_comp(self,x,parms):
        #scl = 1
        return self.scl
    def E(self,x,parms):
        Pm = torch.diag(self.P_d)
        return Pm

class ParametricDiagonal(torch.nn.Module):
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,scl_upper_bound = 0.2,scl_lower_bound = 0.05):
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.parm_dim = parm_dim
        #self.hidden_dim = 10
        self.hidden_dim = np.round(10*self.parm_dim).astype(int)  #THIS IS WHAT I HAVE BEEN USING FOR THE OTHER EXPERIMENTS
        #self.hidden_dim = np.round(2*self.parm_dim).astype(int)

        #self.hidden_dim = np.round(self.parm_dim/2).astype(int)

        self.DiagMap = nn.Sequential(
          nn.Linear(self.parm_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
        self.scl_upper_bound = scl_upper_bound
        self.scl_lower_bound = scl_lower_bound
        self.ScaleMap = nn.Sequential(
                nn.Linear(self.parm_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,1)
                )
    def forward(self,x,parms):
        Pd = self.DiagMap(parms)
        scl = self.scl_lower_bound +  torch.sigmoid(self.ScaleMap(parms))*( self.scl_upper_bound - self.scl_lower_bound)
        P_diag = scl*( self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound) )
        Pm = torch.diag(P_diag)
        return Pm
    def scl_comp(self,x,parms):
        scl = torch.sigmoid(self.ScaleMap(parms))
        return scl

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
    def __init__(self,n_dim,parm_dim,upper_bound,lower_bound,scl_upper_bound = 0.2,scl_lower_bound = 0.05):
        super().__init__()
        self.n_dim = n_dim
        self.P_diag_upper_bound = upper_bound
        self.P_diag_lower_bound = lower_bound
        self.input_dim = parm_dim + n_dim
        #self.hidden_dim = 20
        self.hidden_dim = np.round(10*self.input_dim).astype(int)
        self.DiagMap = nn.Sequential(
          nn.Linear(self.input_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.hidden_dim),
          nn.ReLU(),
          nn.Linear(self.hidden_dim,self.n_dim),
        )
        self.scl_upper_bound = scl_upper_bound
        self.scl_lower_bound = scl_lower_bound
        self.ScaleMap = nn.Sequential(
                nn.Linear(self.input_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim,1)
                )
    def forward(self,x,parms):
        xsp = torch.cat((x,parms),dim = -1)
        Pd = self.DiagMap(xsp)
        scl = self.scl_lower_bound +  torch.sigmoid(self.ScaleMap(xsp))*( self.scl_upper_bound - self.scl_lower_bound)
        P_diag = scl*( self.P_diag_lower_bound + torch.sigmoid(Pd)*(self.P_diag_upper_bound - self.P_diag_lower_bound) )
        Pm = torch.diag(P_diag)
        return Pm
    def scl_comp(self,x,parms):
        xsp = torch.cat((x,parms),dim = -1)
        scl = torch.sigmoid(self.ScaleMap(xsp))
        return scl



