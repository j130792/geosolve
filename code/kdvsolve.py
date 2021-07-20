"""
Here we apply the modified gmres algorithm to lkdv
"""
#Global
import numpy as np

#Local
from geosolve import gmres_e

def kdvsolve(dic,x0,k):

    A = dic['A']
    b = dic['b']
    M = dic['M']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    e0 = dic['e0']

    #Define constraints
    def const1(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out

    def const3(z,x0,Q):
        X = x0 + Q @ z
        out = 0.5 * np.transpose(X) @ L @ X \
            - 0.5 * np.transpose(X) @ M @ X \
            - e0
        return out

    #And stuff them in a list
    conlist = [const1,const3]


    out = gmres_e(A=A,b=b,x0=x0,k=k,
                  conlist=conlist)
    return out


def kdvsolve3(dic,x0,k):

    A = dic['A']
    b = dic['b']
    M = dic['M']
    L = dic['L']
    omega = dic['omega']
    m0 = dic['m0']
    mo0 = dic['mo0']
    e0 = dic['e0']

    #Define constraints
    def const1(z,x0,Q):#Depends on x0 and Q
        X = x0 + Q @ z
        out = np.transpose(omega) @ X - m0
        return out
    
    def const2(z,x0,Q):
        X = x0 + Q @ z
        out = 0.5*np.transpose(X) @ M @ X - mo0
        return out
    
    def const3(z,x0,Q):
        X = x0 + Q @ z
        out = 0.5 * np.transpose(X) @ L @ X \
            - 0.5 * np.transpose(X) @ M @ X \
            - e0
        return out

    #And stuff them in a list
    conlist = [const1,const3,const2]


    out = gmres_e(A=A,b=b,x0=x0,k=k,
                  conlist=conlist)
    return out
