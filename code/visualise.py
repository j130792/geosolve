#global
from firedrake import *
import numpy as np
import pandas as pd
from prettytable import PrettyTable

# local
import lkdv
import refd
import solve

#initialise global lists
## here we will tabulate data
def tabulator(params,prob,dict_list,filename='table'):
    #initalise tables
    t = PrettyTable()
    df = pd.DataFrame()
    #loop over dictionaries (each is a method)
    for data in dict_list:
        name = data['name']
        t.add_column(name + ' res', data['res'])
        df[name + ' residual norm'] = data['res']

        #compute invariants
        dev1 = []
        dev2 = []
        for j in range(1,np.shape(data['x'])[0]):
            inv = lkdv.compute_invariants(prob,data['x'][j])
            dev1.append(inv['mass'] - params['m0'])
            dev2.append(inv['energy'] - params['e0'])

        t.add_column(name + ' m dev', dev1)
        t.add_column(name + ' e dev', dev2)
        df[name + ' mass deviation'] = dev1
        df[name + ' energy deviation'] = dev2


    #write to file
    texfile = open(filename + '.tex', 'w')
    texfile.write(df.to_latex(index=False))
    texfile.close()
    df.to_csv(filename + '.csv', index=False)
        
    print(t)

    return df
        
        
