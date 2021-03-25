#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

#local
import refd

class problem(object):
    def __init__(self,N,M):
        self.mlength = 40
        self.degree = 1
        self.N = N
        self.M = M

    def mesh(self):
        return PeriodicIntervalMesh(self.M,self.mlength)

    def function_space(self,mesh):
        U = FunctionSpace(mesh,"CG",self.degree)
        return MixedFunctionSpace((U,U))

    def exact(self,x,t):
        """
        An "exact" initial condition for the linear KdV equation
        """
        alpha = 4
        period = 2*pi/self.mlength
        beta = alpha*period
        u = sin(beta*(x-(1-beta**2)*t))
        return u

def linforms(N=100,M=50,T=1):
    #set up problem class
    prob = problem(N=N,M=M)
    #Set up finite element stuff
    mesh = prob.mesh()
    Z = prob.function_space(mesh)

    #Set up initial conditions
    z0 = Function(Z)
    u0,v0 = z0.split()

    t = 0.
    x = SpatialCoordinate(Z.mesh())
    u0.assign(project(prob.exact(x[0],t),Z.sub(0)))

    #Define timestep
    dt = float(T)/N

    #Build weak form
    phi, psi = TestFunctions(Z)
    z1 = Function(Z)
    z_trial = TrialFunction(Z)
    u_trial, v_trial = split(z_trial)
    z1.assign(z0)

    u1, v1 = split(z1)
    u0, v0 = split(z0)

    ut = (u_trial - u0) / dt
    umid = 0.5 * (u_trial + u0)

    F1 = (ut + v_trial.dx(0)) * phi * dx
    F2 = (v_trial + umid) * psi * dx \
        + umid.dx(0) * psi.dx(0) * dx
    F = F1 + F2

    #Read out A and b
    A = assemble(lhs(F),mat_type='aij').M.values
    b = np.asarray(assemble(rhs(F)).dat.data).reshape(-1)

    #Get form for mass matrix
    M_form = u_trial * phi * dx
    M = assemble(lhs(M_form),mat_type='aij').M.values
    #And for L
    L_form = u_trial.dx(0) * phi.dx(0) * dx
    L = assemble(lhs(L_form),mat_type='aij').M.values

    #Get the initial values for the invariants
    m0 = assemble(u0*dx)
    e0 = assemble((0.5 * u0.dx(0)**2 - 0.5 * u0**2)*dx)

    #Generate x vector
    u0, v0 = z0.split()
    u0.assign(interpolate(x[0],Z.sub(0)))
    v0.assign(interpolate(x[0],Z.sub(0)))
    x_vec = np.asarray(assemble(z0).dat.data).reshape(-1)


    #Generate output dictionary
    out = {
        'A': A,
        'b': b,
        'x': x_vec,
        'M': M,
        'L': L,
        'm0': m0,
        'e0': e0,
    }
        
    return out, prob


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)

    refd.nptofd(prob,dict['b'])
