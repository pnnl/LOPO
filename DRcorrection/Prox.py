import torch
import numpy as np
import torch.nn as nn
from torch.func import grad
from torch.func import vmap
from torch.func import jacrev
from torch.func import hessian


''' 
#########################

######## PROX DEFINITIONS

##########################

General Equality Constraint
General Box Constraint
General First Order Objective Composed With Equality
General Second Order Objective Composed With Equality

'''


class EqualityConstraint(torch.nn.Module):
    '''
    Approximates the prox operator of the indicator function i_{x : F(x) = 0}
    where F: R^{n}->R^{m}, and m <= n, and is assumed to be differentiable everywhere

    For a given x with J_F(x) the Jacobian of F, the prox approximation is taken to be projection onto the set
    
    { y: F(x) + J_F(x)*(y - x) = 0 }
    
    '''
    def __init__(self,F,JF_fixed = False,n_dim = None, parm_dim = None):
        '''
        :param F:(functorch compatible function) a parameterized function F with input of the form F(x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param n_dim: (int) the dimension of x for precomputing Jacobian
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        '''
        super().__init__()
        self.F = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        self.JF_fixed = JF_fixed
        if self.JF_fixed == True:
            with torch.no_grad():
                jx = torch.zeros((1,n_dim),dtype = torch.float32)
                jp = torch.zeros((1,parm_dim),dtype = torch.float32)
                JFx = self.JF(jx,jp)
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2))
                self.JFx = JFx
                self.Q = Q
                self.R = R
    def forward(self, x, parms):
        if self.JF_fixed == False:
            JFx = self.JF(x,parms)
            ### Take a QR decomposition of the Jacobian
            with torch.no_grad():
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2))
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Q = torch.tile(self.Q,(batch_size,1,1))
            R = torch.tile(self.R,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        R_T = torch.transpose(R,1,2)
        zeta_r = torch.linalg.solve_triangular(R_T,b,upper=False,left = True)
        zeta = torch.bmm(Q,zeta_r)
        ### Compute the Projection Operation
        z = x_mat - zeta
        zq = torch.bmm(torch.transpose(Q,1,2),z)
        zq = torch.bmm(Q,zq)
        z = z- zq 
        x_new = z + zeta
        return torch.squeeze(x_new,dim = -1)

class BoxConstraint(torch.nn.Module):
    '''
    Computes the projection onto the box constraints l_b <= x <= u_b for constants l_b and u_b.
    '''
    def __init__(self,f_lower_bound,f_upper_bound):
        '''
        :param f_lower_bound: (function) has form f(parms), returns tensor of lower bound constraints, is defined unbatched, will be 'raised' with vmap.
        :param f_upper_bound: (function) has form f(parms), returns tensor of upper bound constraints, is defined unbatched, will be 'raised' with vmap.
        '''
        super().__init__()
        self.lower_bound_func = vmap(f_lower_bound)
        self.upper_bound_func = vmap(f_upper_bound)
    def forward(self, x, parms):
        l_b = self.lower_bound_func(parms)
        u_b = self.upper_bound_func(parms)
        return l_b + torch.relu( u_b - l_b - torch.relu(u_b - x ))

class FirstOrderObjectiveConstraintComposition(torch.nn.Module):

    ''' 
    Computes an approximation of the prox operator of the sum
    
    f(x) + i_{ x : F(x) = 0 }

    where F: R^{n}->R^{m}, and m <= n , and is assumed to be differentiable everywhere
          i_{} is the indicator function
          f: R^{n}-> R is scalar valued and assumed to be differentiable everywhere

    
    For a given x the prox operator is computed for the approximation
    
    f(x) + grad_f(x)^T(y - x) + i_{ y: F(x) + J_F(x)*(y - x) = 0 }

    with grad_f(x) the gradient of f at x, and J_F(x), the Jacobian of F at x
    '''
    def __init__(self,f,F,JF_fixed = False,n_dim = None, parm_dim = None):
        '''
        :param f: (functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param F:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param n_dim: (int) the dimension of x for precomputing Jacobian
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        '''
        super().__init__()

        self.f = vmap(f)
        self.f_grad = vmap(grad(f,argnums = 0))
        self.F = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        self.gamma = 2.0
        self.JF_fixed = JF_fixed
        if self.JF_fixed == True:
            with torch.no_grad():
                jx = torch.zeros((1,n_dim),dtype = torch.float32)
                jp = torch.zeros((1,parm_dim),dtype = torch.float32)
                JFx = self.JF(jx,jp)
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2))
                self.JFx = JFx
                self.Q = Q
                self.R = R
    def forward(self, x, parms):
        if self.JF_fixed == False:
            JFx = self.JF(x,parms)
            ### Take a QR decomposition of the Jacobian
            with torch.no_grad():
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2))
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Q = torch.tile(self.Q,(batch_size,1,1))
            R = torch.tile(self.R,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        R_T = torch.transpose(R,1,2)
        zeta_r = torch.linalg.solve_triangular(R_T,b,upper=False,left = True)
        zeta = torch.bmm(Q,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        ### Compute the gradient step and Projection Operation
        z = x_mat - zeta - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Q,1,2),z)
        zq = torch.bmm(Q,zq)
        z = z- zq 
        x_new = zeta + z
        return torch.squeeze(x_new,dim = -1)

class SecondOrderObjectiveConstraintComposition(torch.nn.Module):

    ''' 
    Computes an approximation of the prox operator of the sum
    
    f(x) + i_{ x : F(x) = 0 }

    where F: R^{n}->R^{m}, and m <= n , and is assumed to be differentiable everywhere
          i_{} is the indicator function
          f: R^{n}-> R is scalar valued and assumed to be twice differentiable everywhere

    
    For a given x the prox operator is computed for the approximation
    
    f(x) + grad_f(x)^T(y - x) + (y-x)^T H_f(x) (y-x) + i_{ y: F(x) + J_F(x)*(y - x) = 0 }

    with grad_f(x) the gradient of f at x, H_f(x) the hessian of f at x, and  J_F(x) the Jacobian of F at x

    '''
    def __init__(self,f,F,JF_fixed = False,n_dim = None, parm_dim = None):
        '''
        :param f: (functorch compatible function) a parameterized function f with input of the form (x,parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  f is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param F:(functorch compatible function) a parameterized function F with input of the form (x, parms) where parms is a tensor that matches the batch of x, and has last dim parm features.
                  F is defined unbatched, the method will call vmap, to "raise" f to batch dim

        :param JF_fixed: (Bool) Indicates if the Jacobian of F should be computed at each iteration. Default is False, if True Jacobian of F will be precomputed at x=0, parms = 0
        :param n_dim: (int) the dimension of x for precomputing Jacobian
        :param parm_dim: (int) the dimension of parms for precomputing Jacobian
        '''
        super().__init__()
        self.f = vmap(f)
        self.f_grad = vmap(grad(f,argnums = 0))
        self.H_vec = vmap(hessian(f,argnums = 0))
        self.F = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        #self.gamma = 1e-2
        self.gamma = 2.0
        self.JF_fixed = JF_fixed
        if self.JF_fixed == True:
            with torch.no_grad():
                jx = torch.zeros((1,n_dim),dtype = torch.float32)
                jp = torch.zeros((1,parm_dim),dtype = torch.float32)
                JFx = self.JF(jx,jp)
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Q.shape[-1] - R.shape[-1]
                R = R[:,:-null_dim,:]
                Qr = Q[:,:,:-null_dim]
                Qn = Q[:,:,-null_dim:]
                self.JFx = JFx
                self.Qr = Qr
                self.Qn = Qn
                self.R = R
    def forward(self, x, parms):
        if self.JF_fixed == False:
            JFx = self.JF(x,parms)
            ### Take a QR decomposition of the Jacobian
            with torch.no_grad():
                Q, R  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Q.shape[-1] - R.shape[-1]
                R = R[:,:-null_dim,:]
                Qr = Q[:,:,:-null_dim]
                Qn = Q[:,:,-null_dim:]
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Qr = torch.tile(self.Qr,(batch_size,1,1))
            Qn = torch.tile(self.Qn,(batch_size,1,1))
            R = torch.tile(self.R,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        R_T = torch.transpose(R,1,2)
        zeta_r = torch.linalg.solve_triangular(R_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        # Compute the Hessian of f
        Hf = self.H_vec(x,parms)
        ### Compute the gradient step and Projection Operation
        batch_size = x.shape[0]
        n_dim = x.shape[-1]
        Id = torch.tile(torch.unsqueeze(torch.eye(n_dim),dim=0),(batch_size,1,1))
        Md =  (self.gamma/2)*Id + Hf
        QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
        zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)
        zq = torch.linalg.solve(QTMdQ,zq)
        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta
        return torch.squeeze(x_new,dim = -1)















