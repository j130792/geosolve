#global
from firedrake import *
import numpy as np
from tabulate import tabulate
from prettytable import PrettyTable

# local
import lkdv
import refd
import solve

#initialise global lists

headers = []
t = PrettyTable()
## here we will tabulate data
def tabulator(params,prob,dict_list):
    #loop over dictionaries (each is a method)
    for data in dict_list:
        name = data['name']
        t.add_column(name + ' res', data['res'])

        #compute invariants
        dev1 = []
        dev2 = []
        for j in range(1,np.shape(data['x'])[0]):
            inv = lkdv.compute_invariants(prob,data['x'][j])
            dev1.append(inv['mass'] - params['m0'])
            dev2.append(inv['energy'] - params['e0'])

        t.add_column(name + ' m dev', dev1)
        t.add_column(name + ' e dev', dev2)


    print(t)

    return t
        
        
