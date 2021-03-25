#global
from firedrake import *
import numpy as np
#local
import lkdv


def gmres(A, b, x0, k):

    x = []
    r = (b - np.dot(A,x0)) #define r0

    x.append(r)

         
    q = [0] * k #initalise stuff

    beta = np.linalg.norm(r)
    
    q[0] = r / beta #normalise r0
    
    #convert type
    
    h = np.zeros((k+1,k))    
    
    for j in range(k):
        y = np.asarray(np.dot(A,q[j]))
        
        for i in range(j):
            h[i,j] = np.dot(q[i],y)
            y = y - h[i,j] * q[i]
        h[j+1,j] = np.linalg.norm(y)
        if (h[j+1,j] != 0 and j!=k-1):
            q[j+1] =  y / h[j+1,j]

        res = np.zeros(k+1)
        res[0] = beta

        
        yk = np.linalg.lstsq(h, res)[-1]
        print(yk)

        
        x.append(np.dot(np.transpose(q),yk) + x0)
    
    
    print(x)
    


    return -1



if __name__=="__main__":

    params, prob = lkdv.linforms()

    print(gmres(params['A'],
                params['b'],
                x0=np.ones_like(params['b']),
                k=4
                )
          )
    
