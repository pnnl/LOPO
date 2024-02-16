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
General Bound Constraints
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
        # Find a solution to RHS
        R_T = torch.transpose(R,1,2)
        zeta_r = torch.linalg.solve_triangular(R_T,b,upper=False,left = True)
        zeta = torch.bmm(Q,zeta_r)
        # Compute the Projection Operation
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
    def __init__(self,f,F,metric,JF_fixed = False,n_dim = None, parm_dim = None):
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
        self.metric = metric
        self.JF_fixed = JF_fixed
        if self.JF_fixed == True:
            jx = torch.zeros((1,n_dim),dtype = torch.float32)
            jp = torch.zeros((1,parm_dim),dtype = torch.float32)
            JFx = self.JF(jx,jp)
            Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Qc.shape[-1] - Rc.shape[-1]
            Rr = Rc[:,:-null_dim,:]
            Qr = Qc[:,:,:-null_dim]
            Qn = Qc[:,:,-null_dim:]
            self.JFx = JFx
            self.Qr = Qr
            self.Qn = Qn
            self.Rr = Rr
    def forward(self, x, parms):
        if self.JF_fixed == False:
            with torch.no_grad():
                JFx = self.JF(x,parms)
                ### Take a QR decomposition of the Jacobian
                Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Qc.shape[-1] - Rc.shape[-1]
                Rr = Rc[:,:-null_dim,:]
                Qr = Qc[:,:,:-null_dim]
                Qn = Qc[:,:,-null_dim:]
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Qr = torch.tile(self.Qr,(batch_size,1,1))
            Qn = torch.tile(self.Qn,(batch_size,1,1))
            Rr = torch.tile(self.Rr,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        ### Compute the gradient step and Projection Operation
        Pm = vmap(self.metric)(x,parms)
        QtPQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Pm),Qn)
        zq = torch.bmm(Pm,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)
        zq = torch.linalg.solve(QtPQ,zq)
        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta
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
    def __init__(self,f,F,metric,JF_fixed = False,n_dim = None, parm_dim = None,gamma = None):
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
        if gamma !=None:
            self.gamma = gamma
        else:
            self.gamma = 2.0
        self.metric = metric
        self.JF_fixed = JF_fixed
        if self.JF_fixed == True:
            jx = torch.zeros((1,n_dim),dtype = torch.float32)
            jp = torch.zeros((1,parm_dim),dtype = torch.float32)
            JFx = self.JF(jx,jp)
            Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
            null_dim = Qc.shape[-1] - Rc.shape[-1]
            Rr = Rc[:,:-null_dim,:]
            Qr = Qc[:,:,:-null_dim]
            Qn = Qc[:,:,-null_dim:]
            self.JFx = JFx
            self.Qr = Qr
            self.Qn = Qn
            self.Rr = Rr
    def forward(self, x, parms):
        #print(self.metric.P_d)
        if self.JF_fixed == False:
            with torch.no_grad():
                JFx = self.JF(x,parms)
                ### Take a QR decomposition of the Jacobian
                Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
                null_dim = Qc.shape[-1] - Rc.shape[-1]
                Rr = Rc[:,:-null_dim,:]
                Qr = Qc[:,:,:-null_dim]
                Qn = Qc[:,:,-null_dim:]
        if self.JF_fixed == True:
            batch_size = x.shape[0]
            JFx = torch.tile(self.JFx,(batch_size,1,1))
            Qr = torch.tile(self.Qr,(batch_size,1,1))
            Qn = torch.tile(self.Qn,(batch_size,1,1))
            Rr = torch.tile(self.Rr,(batch_size,1,1))
        Fx = self.F(x,parms)
        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)
        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)
        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)
        # Compute the Hessian of f
        Hf = self.H_vec(x,parms)
        ### Compute the gradient step and Projection Operation
        Pm = vmap(self.metric)(x,parms)
        Md =  (self.gamma/2)*Pm + Hf
        QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
        zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)
        zq = torch.linalg.solve(QTMdQ,zq)
        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta
        return torch.squeeze(x_new,dim = -1)




# An adaptation of SecondOrderObjectiveConstraintComposition,
# to the special case where the optimization problem is already a QP
# The main point will be to take advantage of opportunities to increase efficiency,
#   based on assuming the objective and constraint 'approximations' to be
#   constant, and using prefactorizations
# However, that requires modularizing the forward pass into a fwd and precompute phase,
#   which doesn't match the torch.nn.Module pattern.
# This class does not implement precomputes, and serves mainly to benchmark and compare against
#   the working class ModularQuadraticObjectiveLinearEqualityComposition

class QuadraticObjectiveLinearEqualityComposition(torch.nn.Module):
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

    def __init__(self,f,F,metric,n_dim = None, parm_dim = None):
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

        self.gamma = 2.0

        self.metric = metric

        #self.JF_fixed = JF_fixed
        #if self.JF_fixed == True:
        # The constraints Jacobian is always fixed
        jx = torch.zeros((1,n_dim),dtype = torch.float32)
        jp = torch.zeros((1,parm_dim),dtype = torch.float32)
        JFx = self.JF(jx,jp)
        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        null_dim = Qc.shape[-1] - Rc.shape[-1]
        Rr = Rc[:,:-null_dim,:]
        Qr = Qc[:,:,:-null_dim]
        Qn = Qc[:,:,-null_dim:]

        self.JFx = JFx
        self.Qr = Qr
        self.Qn = Qn
        self.Rr = Rr

        self.Hf = self.H_vec(jx,jp) # this needs a dummy input


    def forward(self, x, parms):
        #print(self.metric.P_d)

        #if self.JF_fixed == False:
        #    with torch.no_grad():
        #        JFx = self.JF(x,parms)
        #        ### Take a QR decomposition of the Jacobian
        #        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        #        null_dim = Qc.shape[-1] - Rc.shape[-1]
        #        Rr = Rc[:,:-null_dim,:]
        #        Qr = Qc[:,:,:-null_dim]
        #        Qn = Qc[:,:,-null_dim:]

        #if self.JF_fixed == True:
        batch_size = x.shape[0]
        JFx = torch.tile(self.JFx,(batch_size,1,1))
        Qr  = torch.tile(self.Qr, (batch_size,1,1))
        Qn  = torch.tile(self.Qn, (batch_size,1,1))
        Rr  = torch.tile(self.Rr, (batch_size,1,1))

        Fx = self.F(x,parms)

        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)   # can we just do a matmul on the non-unsqueezed x? Would it even be faster?

        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)


        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)

        # Compute the Hessian of f
        #Hf = self.H_vec(x,parms)
        Hf = self.Hf

        ### Compute the gradient step and Projection Operation
        Pm = vmap(self.metric)(x,parms)
        Md =  (self.gamma/2)*Pm + Hf
        QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
        # Factorize QTMdQ


        zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)


        Q,R = torch.linalg.qr(QTMdQ)
        zq = torch.linalg.solve_triangular(R,torch.bmm(torch.transpose(Q,1,2),zq),upper=True)
        #zq = torch.linalg.solve(QTMdQ,zq)

        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta

        return torch.squeeze(x_new,dim = -1)



# A modular version which predicts the metric and factorizes the quadratic
#  prox problem only once per solver call, rather than once per iteration.
class ModularQuadraticObjectiveLinearEqualityComposition():


    def __init__(self,f,F,metric,n_dim = None, parm_dim = None):
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

        self.gamma = 2.0

        self.metric = metric

        #self.JF_fixed = JF_fixed
        #if self.JF_fixed == True:
        # The constraints Jacobian is always fixed
        jx = torch.zeros((1,n_dim),dtype = torch.float32)
        jp = torch.zeros((1,parm_dim),dtype = torch.float32)
        JFx = self.JF(jx,jp)
        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        null_dim = Qc.shape[-1] - Rc.shape[-1]
        Rr = Rc[:,:-null_dim,:]
        Qr = Qc[:,:,:-null_dim]
        Qn = Qc[:,:,-null_dim:]

        self.JFx = JFx#.squeeze(0)
        self.Qr = Qr
        self.Qn = Qn
        self.Rr = Rr

        self.Hf = self.H_vec(jx,jp) # this needs a dummy input


    def prox(self, x, Q, R, Md, parms):
        #print(self.metric.P_d)

        #if self.JF_fixed == False:
        #    with torch.no_grad():
        #        JFx = self.JF(x,parms)
        #        ### Take a QR decomposition of the Jacobian
        #        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        #        null_dim = Qc.shape[-1] - Rc.shape[-1]
        #        Rr = Rc[:,:-null_dim,:]
        #        Qr = Qc[:,:,:-null_dim]
        #        Qn = Qc[:,:,-null_dim:]

        #if self.JF_fixed == True:
        batch_size = x.shape[0]
        JFx = torch.tile(self.JFx,(batch_size,1,1))
        Qr  = torch.tile(self.Qr, (batch_size,1,1))
        Qn  = torch.tile(self.Qn, (batch_size,1,1))
        Rr  = torch.tile(self.Rr, (batch_size,1,1))



        Fx = self.F(x,parms)

        # Compute the RHS
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)   # can we just do a matmul on the non-unsqueezed x? Would it even be faster?


        #### Find a solution to RHS
        Rr_T = torch.transpose(Rr,1,2)
        zeta_r = torch.linalg.solve_triangular(Rr_T,b,upper=False,left = True)
        zeta = torch.bmm(Qr,zeta_r)


        # Compute the gradient of f
        fg = torch.unsqueeze(self.f_grad(x,parms),dim = -1)

        # Compute the Hessian of f
        #Hf = self.H_vec(x,parms)
        Hf = self.Hf

        ### Compute the gradient step and Projection Operation
        #Pm = vmap(self.metric)(x,parms)
        #Md =  (self.gamma/2)*Pm + Hf
        #QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)
        zq = torch.bmm(Md,x_mat - zeta) - (self.gamma/2)*fg
        zq = torch.bmm(torch.transpose(Qn,1,2),zq)

        #Q,R = torch.linalg.qr(QTMdQ)
        zq = torch.linalg.solve_triangular(R,torch.bmm(torch.transpose(Q,1,2),zq),upper=True)
        #zq = torch.linalg.solve(QTMdQ,zq)

        zq = torch.bmm(Qn,zq)
        x_new =  zq + zeta

        return torch.squeeze(x_new,dim = -1)



    def prefactor(self, x, parms):
        #print(self.metric.P_d)

        #if self.JF_fixed == False:
        #    with torch.no_grad():
        #        JFx = self.JF(x,parms)
        #        ### Take a QR decomposition of the Jacobian
        #        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        #        null_dim = Qc.shape[-1] - Rc.shape[-1]
        #        Rr = Rc[:,:-null_dim,:]
        #        Qr = Qc[:,:,:-null_dim]
        #        Qn = Qc[:,:,-null_dim:]

        #if self.JF_fixed == True:
        batch_size = x.shape[0]

        Qn  = torch.tile(self.Qn, (batch_size,1,1))

        # Compute the Hessian of f
        #Hf = self.H_vec(x,parms)
        Hf = self.Hf

        ### Compute the gradient step and Projection Operation
        Pm = vmap(self.metric)(x,parms)
        Md =  (self.gamma/2)*Pm + Hf
        QTMdQ = torch.bmm(torch.bmm(torch.transpose(Qn,1,2),Md),Qn)

        Q,R = torch.linalg.qr(QTMdQ)

        return Q,R,Md











class EqualityConstrainedQuadratic(torch.nn.Module):


    def __init__(self,f,F,metric,JF_fixed = False,n_dim = None, parm_dim = None, slack_mode = False, n_ineq = 0):
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

        self.F  = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))

        self.gamma = 1.0

        self.metric = metric

        self.slack_mode = slack_mode   # JK new to this class - can these be inferred from the orgininal arguments?
        self.n_ineq = n_ineq


        #vHf  = vmap(hessian(f,argnums = 0))
        #z = 2*torch.ones(3,5)  # a batch of inputs to the prox
        #p = 2*torch.ones(3,5)  # a batch of corresponding problem parameters

        #Hfx = vHf( z,p )

        #self.JF_fixed = JF_fixed
        #if self.JF_fixed == True:
        # The constraints Jacobian is always fixed
        #jx = torch.zeros((1,n_dim),dtype = torch.float32)
        #jp = torch.zeros((1,parm_dim),dtype = torch.float32)
        #JFx = self.JF(jx,jp)
        #Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        #null_dim = Qc.shape[-1] - Rc.shape[-1]
        #Rr = Rc[:,:-null_dim,:]
        #Qr = Qc[:,:,:-null_dim]
        #Qn = Qc[:,:,-null_dim:]

        #self.JFx = JFx
        #self.Qr = Qr
        #self.Qn = Qn
        #self.Rr = Rr

        #self.Hf = self.H_vec(jx,jp) # this needs a dummy input


    def forward(self, x, parms):
        #print(self.metric.P_d)

        #if self.JF_fixed == False:
        #    with torch.no_grad():
        #        JFx = self.JF(x,parms)
        #        ### Take a QR decomposition of the Jacobian
        #        Qc, Rc  = torch.linalg.qr(torch.transpose(JFx,1,2),mode = 'complete')
        #        null_dim = Qc.shape[-1] - Rc.shape[-1]
        #        Rr = Rc[:,:-null_dim,:]
        #        Qr = Qc[:,:,:-null_dim]
        #        Qn = Qc[:,:,-null_dim:]

        #if self.JF_fixed == True:
        batch_size = x.shape[0]
        n_dim = x.shape[1]
        parm_dim = parms.shape[1]
        z = x

        if self.slack_mode:
            p = torch.cat( (parms,torch.zeros(batch_size,self.n_ineq)),1 )   # a hack
        else:
            p = parms

        #Fx  =  vF( z,p )
        #JFx = vJF( z,p )
        #Hfx = vHf( z,p )
        #F0  =  vF(jx,jp)

        #print("p = ")
        #print( p    )
        #print("p.shape = ")
        #print( p.shape    )

        jx = torch.zeros((batch_size,n_dim),dtype = torch.float32)
        jp = torch.zeros((batch_size,parm_dim),dtype = torch.float32)

        Fx  = self.F( z,p )
        JFx = self.JF( z,p )
        Hfx = self.H_vec( z,p )
        F0  = self.F(jx,jp)


        #Pm = vmap(self.metric)(x,parms)
        #P = Pm
        # JK TODO: replace with predicted metric
        P = torch.eye(n_dim).repeat(batch_size,1,1)
        #P = torch.zeros(n_dim,n_dim).repeat(batch_size,1,1)

        Q = Hfx/2.0
        A = JFx
        b = -F0  # A hack since it assumes F is linear
        O = torch.zeros(batch_size,A.shape[1],A.shape[1])

        #print("b = ")
        #print( b    )

        KKTtop = torch.cat( (Q+(1.0/(self.gamma))*P,A.permute(0,2,1)),2 )
        KKTbot = torch.cat( (A,O),2   )
        KKT = torch.cat( (KKTtop,KKTbot),1 )

        #print("KKT = ")
        #print( KKT    )
        #input("waiting")

        rhstop = -p + torch.bmm((1/self.gamma)*P,z.unsqueeze(2)).squeeze(2)
        rhsbot = b

        rhs = torch.cat( (rhstop,rhsbot),1 )

        sol = torch.linalg.solve(KKT,rhs)[:,:n_dim]

        return sol












class ADMMEqualityConstrainedQuadratic(torch.nn.Module):
    def __init__(self,f,F,metric,n_dim,JF_fixed = False, parm_dim = None, n_ineq = 0, E = None):
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
        self.F  = vmap(F)
        self.JF = vmap(jacrev(F,argnums = 0))
        self.gamma = 1.0
        self.metric = metric
        self.n_ineq = n_ineq
        if E !=None: 
            self.E = E 
        else: 
            self.E = torch.eye(n_dim)
        
        self.P = torch.matmul(torch.transpose(E),E)
    def forward(self, x, parms):
        batch_size = x.shape[0]
        n_dim = x.shape[1]
        parm_dim = parms.shape[1]
        Fx  = self.F(x,parms)
        JFx = self.JF(x,parms)
        Hfx = self.H_vec(x,parms)
        # Compute the RHS of constraint approximation 
        Fx_mat = torch.unsqueeze(Fx,dim = -1)
        x_mat = torch.unsqueeze(x,dim = -1)
        b = -Fx_mat + torch.bmm(JFx,x_mat)

        p = self.f_grad(x,parms)

        #Pm = vmap(self.metric)(x,parms)
        #P = Pm
        # JK TODO: replace with predicted metric
        #P = torch.eye(n_dim).repeat(batch_size,1,1)
        P_batch = self.P.repeat(batch_size,1,1)
        #P = torch.zeros(n_dim,n_dim).repeat(batch_size,1,1)

        Q = Hfx/2.0
        A = JFx
        #b = -F0  # A hack since it assumes F is linear
        O = torch.zeros(batch_size,A.shape[1],A.shape[1])

        #print("b = ")
        #print( b    )

        KKTtop = torch.cat( (Q+(2*self.gamma)*P_batch,A.permute(0,2,1)),2 )
        KKTbot = torch.cat( (A,O),2   )
        KKT = torch.cat( (KKTtop,KKTbot),1 )

        #print("KKT = ")
        #print( KKT    )
        #input("waiting")

        rhstop = -p + torch.bmm((1/self.gamma)*P_batch,x.unsqueeze(2)).squeeze(2)
        rhsbot = b

        rhs = torch.cat( (rhstop,rhsbot),1 )

        sol = torch.linalg.solve(KKT,rhs)[:,:n_dim]

        return sol
