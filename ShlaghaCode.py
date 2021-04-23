# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 10:11:15 2021

@author: Shlagha
"""

# matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import integrate
import MyTicToc as mt
import pandas as pd

# Definition of parameters
a = 1
b = 0.25
c = 0.1
d = 0.01


# Definition of Rate Equation
def dYdt(t, Y):
    """ Return the growth rate of fox and rabbit populations. """
    return np.array([a*Y[0] - b*Y[0]*Y[1],
                     -c*Y[1] + d*Y[0]*Y[1]])


def main():
    # Definition of output times
    tOut = np.linspace(0, 100, 200)              # time
    nOut = np.shape(tOut)[0]

    # Initial case, 10 rabbits, 5 foxes
    Y0 = np.array([10, 5])
    mt.tic()
    t_span = [tOut[0], tOut[-1]]
    YODE = sp.integrate.solve_ivp(dYdt, t_span, Y0, t_eval=tOut, 
                                  method='RK45', vectorized=True, 
                                  rtol=1e-5 )
    # infodict['message']                     # >>> 'Integration successful.'
    rODE = YODE.y[0,:]
    fODE = YODE.y[1,:]
    
    # Plot results with matplotlib
    plt.figure()
    plt.plot(tOut, rODE, 'r-', label='RODE')
    plt.plot(tOut, fODE, 'b-', label='FODE')
    
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time')
    plt.ylabel('population')
    plt.title('Evolution of fox and rabbit populations')
    # f1.savefig('rabbits_and_foxes_1.png')
    plt.show()
    
    plt.figure()
    plt.plot(fODE, rODE, 'b-', label='ODE')
    
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Foxes')    
    plt.ylabel('Rabbits')
    plt.title('Evolution of fox and rabbit populations')
    # f2.savefig('rabbits_and_foxes_2.png')
    plt.show()


if __name__ == "__main__":
    main()
    
    #%%
#==========================================================================
'Assignment 1'

#definition of parameters
beta0 = 1
a = 1
bcl = 1
bwb = 1
Cf = 1
Scl_min = 0 #m3
Scl_max = 0.3*(9100+1.5/13.5*(28355-9100))/2*1.5 #m3
Sev_min = 0.000*9100 #m3 #assumed min evap 
Sev_max = 0.0057*9100 #m3 #assumed max evap from the excel 

#the base area of the cover
#Wb = 

print('')
print('Scl_min is: ', Scl_min)
print('Scl_max is: ', Scl_max)
print('Sev_min is: ', Sev_min)
print('Sev_max is: ', Sev_max)
print('')
#%%
# Defining the data
data = pd.read_excel('G7.xlsx')

Qdr = data['Q'].iloc[:]
Jrf = data['Rain'].iloc[:]
pEV = data['pEV'].iloc[:]
Temp = data['Temp'].iloc[:]


#%%
#Define the equation



#def Lcl(a, Scl, Scl_min, Scl_max, bcl):
#Lcl = a*( (Scl - Scl_min)/(Scl_max - Scl_min) )**bcl
    #return Lcl

def dSdt():
    #How to find Scl and Swb ???
    Scl = 8500 #m3 ???????
    Swb = 9000 #m3 ?????
    Lcl = a*( (Scl - Scl_min)/(Scl_max - Scl_min) )**bcl
    Lwb = a*( (Swb - Swb))
    
    #E(t) Equation:
    if Scl < Sev_min:
        fred = 0
    elif (Sev_min < Scl < Sev_max):
        fred = (Scl - Sev_min)/(Sev_max - Sev_min)
    else:           #for Scl > Sev_max:
        fred = 1
    print(fred)
    E = pEV*Cf*fred