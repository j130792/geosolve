#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

def exact(x,t):
    """
    An "exact" initial condition for the linear KdV equation
    """
    alpha = 4
    period = 2*pi/40
    beta = alpha*period
    u = sin(beta*(x-(1-beta**2)*t))
    return u

def linforms(N=100,M=50,T=1):
    #Set up finite element stuff
    mesh = PeriodicIntervalMesh(M,40)
    U = FunctionSpace(mesh,"CG",1)
    Z = MixedFunctionSpace((U,U))

    #Set up initial conditions
    z0 = Function(Z)
    u0,v0 = z0.split()

    t = 0.
    x = SpatialCoordinate(Z.mesh())
    u0.assign(project(exact(x[0],t),U))

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
    b = assemble(rhs(F)).dat.data

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
    u0.assign(interpolate(x[0],U))
    v0.assign(interpolate(x[0],U))
    x_vec = assemble(z0).dat.data

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
        
    return out


if __name__=="__main__":
    print(linforms())
