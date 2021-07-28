'''
A script of functions to put objects back into Firedrake data structures (2D)
'''
#global
from firedrake import *
import numpy as np
import matplotlib.pylab as plt


def nptofd(prob,vec):

    #Set up spaces
    m = prob.mesh
    Z = prob.function_space(m)

    #Initialise firedrake function
    z = Function(Z)
    u, v = z.split()
    
    #Get sizes of each dimension
    dim1 = len(z.sub(0).dat.data)
    dim2 = len(z.sub(1).dat.data)

    #Split external data
    vec1 = vec[:dim1]
    vec2 = vec[dim1:]

    #Copy in external data
    u.dat.data[:] = vec1[:]
    v.dat.data[:] = vec2[:]
    
    return z


#Combine 2D Firedrake vectors
def combine(vec):

    dim = len(vec)
    size = []
    for i in range(dim):
        size.append(len(vec[i]))

    totalsize = np.sum(size)

    newvec = np.zeros((totalsize,))

    total = 0
    for i in range(dim):
        newvec[total:total+size[i]] = vec[i]
        total = total + size[i]
    
    return newvec
