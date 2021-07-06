#Global imports
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

#local
import refd

class problem(object):
    def __init__(self,N,M,space,degree):
        self.mlength = 40
        self.degree = degree
        self.N = N
        self.M = M
        self.space = space
        self.mesh = PeriodicIntervalMesh(self.M,self.mlength)

    def function_space(self,mesh):
        U = FunctionSpace(mesh,self.space,self.degree)
        return MixedFunctionSpace((U,U))

    def exact(self,x,t):
        """
        An "exact" initial condition for the linear KdV equation
        """
        alpha = 4
        period = 2*pi/self.mlength
        beta = alpha*period
        u = sin(beta*(x-(1-beta**2)*t)) + 1
        return u
        

def linforms(N=100,M=50,degree=1,T=1,space='CG'):
    #set up problem class
    prob = problem(N=N,M=M,space=space,degree=degree)
    #Set up finite element stuff
    mesh = prob.mesh
    Z = prob.function_space(mesh)

    #set up DG stuff
    n = FacetNormal(mesh)
    h = prob.mlength/M
    sigma = 10/h
    #including some definitions
    def bilin(uh,vh):
        a = uh.dx(0)*vh.dx(0)*dx
        b = - (jump(uh,n[0])*avg(vh.dx(0)) 
               +jump(vh,n[0])*avg(uh.dx(0)) ) *dS
        d = (sigma) * jump(uh,n[0])*jump(vh,n[0]) *dS
        return a + b + d

    def gfunc(uh,vh):
        g = uh.dx(0)*vh*dx - jump(uh,n[0])*avg(vh)*dS 
        return g
    
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

    #u1, v1 = split(z1)
    u0, v0 = split(z0)

    ut = (u_trial - u0) / dt
    umid = 0.5 * (u_trial + u0)

    F1 = (ut) * phi * dx + gfunc(v_trial,phi)
    F2 = (v_trial - umid) * psi * dx \
        + bilin(umid,psi)
    F = F1 + F2

    
    #Read out A and b
    A = assemble(lhs(F),mat_type='aij').M.values
    b = np.asarray(assemble(rhs(F)).dat.data).reshape(-1)

    #Get form for mass matrix
    M_form = u_trial * phi * dx
    M = assemble(lhs(M_form),mat_type='aij').M.values
    #And for L
    L_form = bilin(u_trial, phi)
    L = assemble(lhs(L_form),mat_type='aij').M.values
    #And the vector needed for finding the mass
    omega = np.asarray(assemble(phi * dx).dat.data).reshape(-1)

    #Get the initial values for the invariants
    m0 = assemble(u0*dx)
    mo0 = assemble(u0*u0*dx)
    e0 = assemble(0.5*bilin(u0,u0) + ( - 0.5 * u0**2)*dx)

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
        'omega': omega,
        'm0': m0,
        'mo0': mo0,
        'e0': e0,
    }
        
    return out, prob


def compute_invariants(prob,uvec):

    #set up DG stuff
    n = FacetNormal(prob.mesh)
    h = prob.mlength/prob.M
    sigma = 10/h
    #including some definitions
    def bilin(uh,vh):
        a = uh.dx(0)*vh.dx(0)*dx
        b = - (jump(uh,n[0])*avg(vh.dx(0)) 
               +jump(vh,n[0])*avg(uh.dx(0)) ) *dS
        d = (sigma) * jump(uh,n[0])*jump(vh,n[0]) *dS
        return a + b + d
    
    z = refd.nptofd(prob,uvec)
    u,v = z.split()
    mass = assemble(u*dx)
    momentum = assemble(u**2*dx)
    energy = assemble(0.5 * bilin(u,u) - ( 0.5 * u**2)*dx)

    inv_dict = {'mass' : mass,
                'momentum' : momentum,
                'energy' : energy}

    return inv_dict


if __name__=="__main__":
    dict, prob = linforms()
    print(dict)

    refd.nptofd(prob,dict['b'])
