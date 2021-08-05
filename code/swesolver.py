"""
Here we apply the modified gmres algorithm to lkdv
"""
#Global
import numpy as np

#Local
from geosolve import gmres_e

def swesolver(dic,x0,k):

    A = dic['A']
    b = dic['b']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    e0 = dic['e0']

    
    #Define constraints
    def const_mass(z,x0,Q):
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out
    
    def const_energy(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = 0.5 * X @ L @ X - e0
        return out


    #And stuff them in a list
    conlist = [const_mass,const_energy]


    out = gmres_e(A=A,b=b,x0=x0,k=k,
                  conlist=conlist)
    return out

