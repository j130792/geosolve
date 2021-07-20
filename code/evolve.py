"""
Study solution evolution
"""
#global
from firedrake import *
import numpy as np
import matplotlib.pylab as plt

#local
from kdvsolve import kdvsolve
import lkdv
import refd

def kdv2(N=100,M=50,degree=1,k=10):
    #Get linear forms and Firedrake problem class
    forms, prob = lkdv.linforms(N=N,M=M,degree=degree)
    mesh = prob.mesh
    U = prob.function_space(mesh)

    #Initialise solution array
    sol = []
    
    #solve first step
    z, _ = kdvsolve(forms, x0=np.zeros_like(forms['b']), k=10)
    z0 = refd.nptofd(prob,z)
    sol.append(z0)

    #Iteratively solve the remaining steps
    for i in range(1,N):
        #Update forms
        forms, _ = lkdv.linforms(N=N,M=M,degree=degree,zinit=sol[-1])
        #Initial guess at previous solution
        x0 = refd.flatten(sol[-1].dat.data)
        #Solve
        z, _ = kdvsolve(forms, x0=np.zeros_like(forms['b']), k=10)
        #Convert back to FD
        z0 = refd.nptofd(prob,z)
        #Append
        sol.append(z0)
    


    return -1



if __name__=="__main__":
    print(kdv2())
    
    
    
