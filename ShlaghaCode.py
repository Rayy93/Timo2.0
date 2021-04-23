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

#%%

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
ncl = 0.3 #porosity for the cover layer
nwb = 0.3 #porosity for the waste body

#the base area of the cover
H = 1.5 + 12 #m Total height (cover + body)
Acb = 9100 + 1.5/H * (28355-9100)  #m2 (the area of the base of the cover)
Vcl = (9100 + Acb)/2*1.5 #m3 (volume of the cover)
Vwb = (28355 + Acb)/2*12 #m3 (volume of the waste body)

#for the cover:
Scl_min = 0 #m3, min. water content of the cover layer (cl)
Scl_max = ncl*Vcl #m3, max. water content of the cover layer (cl)

#for the evaporation
Sev_min = 0.000*9100 #m3 #assumed min evap 
Sev_max = 0.0057*9100 #m3 #assumed max evap from the excel 

#for the waste body
Swb_min = 0 #m3, min. water content of the waste body (wb)
Swb_max = nwb*Vwb #m3, maximum water content of the wb (porosity x volume)



print('')
print('volume of the cover layer is :', Vcl, 'm3')
print('volume of the waste body layer is :', Vwb, 'm3')
print('')
print('Scl_min is: ', Scl_min, 'm3')
print('Scl_max is: ', Scl_max, 'm3')
print('Sev_min is: ', Sev_min, 'm3')
print('Sev_max is: ', Sev_max, 'm3')
print('Swb_min is: ', Swb_min, 'm3')
print('Swb_max is: ', Swb_max, 'm3')
print('')
#%%

# Defining the data from the excel
data = pd.read_excel('G7.xlsx')

Qdr = data['Q'].iloc[:] #m3/day
Jrf = data['Rain'].iloc[:] #m/day
pEV = data['pEV'].iloc[:] #m/day
Temp = data['Temp'].iloc[:] #celcius

#'Scl and Swb'
#How to find Scl and Swb ???
#Scl = np.linspace(Scl_min, Scl_max, len(Qdr))   #(Scl_min + Scl_max)/2 #m3 initial Scl ???????
#Swb = np.linspace(Swb_min, Swb_max, len(Qdr))   #(Swb_min + Swb_max)/2 #m3 initial Swb ?????
#Lcl = a*( (Scl - Scl_min)/(Scl_max - Scl_min) )**bcl
#Lwb = a*( (Swb - Swb_min)/(Swb_max - Swb_min))**bcl

#%%
#Define the equation



#def Lcl(a, Scl, Scl_min, Scl_max, bcl):
#Lcl = a*( (Scl - Scl_min)/(Scl_max - Scl_min) )**bcl
    #return Lcl


def dSdt(t, Y):    #Y[0] = Scl, Y[1] = Swb
 
    #E(t) Equation:
    #if Y < Sev_min:
    #    fred = 0
    #elif (Sev_min < Y < Sev_max):
    #    fred = (Y - Sev_min)/(Sev_max - Sev_min)
    #else:           #for Scl > Sev_max:
    #    fred = 1
    #print('fred is: ', fred)
    
    #E = pEV * Cf * fred
    #assuming Sev_min < Y < Sev_max
    E = pEV * Cf * (Y[0] - Sev_min)/(Sev_max - Sev_min)
    
    
    #beta = beta0*((Scl - Scl_min)/(Scl_max - Scl_min))
    
    return np.array([Jrf - a*( (Y[0] - Scl_min)/(Scl_max - Scl_min) )**bcl - E,
                   (1-beta0*((Y[0] - Scl_min)/(Scl_max - Scl_min))) * a*( (Y[0] - Scl_min)/(Scl_max - Scl_min) )**bcl - a*( (Y[1] - Swb_min)/(Swb_max - Swb_min))**bwb])
                    #beta0*((Y[0] - Scl_min)/(Scl_max - Scl_min)) * a*( (Y[0] - Scl_min)/(Scl_max - Scl_min) )**bcl + a*( (Y[1] - Swb_min)/(Swb_max - Swb_min))**bwb - Qdr[0]])
                    #the third equation is ignored since it's = 0 (used to find the correct assumption of beta0)
    
    #return np.array([Jrf[0] - a*( (Scl[0] - Scl_min)/(Scl_max - Scl_min) )**bcl - E,
    #               (1-beta0*((Scl[0] - Scl_min)/(Scl_max - Scl_min))) * a*( (Scl[0] - Scl_min)/(Scl_max - Scl_min) )**bcl - a*( (Swb[0] - Swb_min)/(Swb_max - Swb_min))**bwb,
    #                beta0*((Scl[0] - Scl_min)/(Scl_max - Scl_min)) * a*( (Scl[0] - Scl_min)/(Scl_max - Scl_min) )**bcl + a*( (Swb[0] - Swb_min)/(Swb_max - Swb_min))**bwb - Qdr[0]])
    
    #return np.array([Jrf[0] - Lcl[0] - E,
    #               (1-beta)*Lcl[0] - Lwb[0],
    #                beta[0]*Lcl[0] + Lwb[0] - Qdr[0]])


def main2():
    #defition of output times
    tOut2 = np.linspace(0, len(Qdr), len(Qdr)) #"tOut2" to differentiate with what Timo did
    nOut2 = np.shape(tOut2)[0]
    
    #initial case for Scl and Swb ---> Y0 = (Scl[0], Swb[0])
    Y0 = np.array([(Scl_min + Scl_max)/2 , (Swb_min + Swb_max)/2 ])
    mt.tic() #to start the timer, to give calculation time record
    t_span = [tOut2[0], tOut2[-1]]
    YODE2 = sp.integrate.solve_ivp(dSdt, t_span, Y0, t_eval=tOut2, 
                                   method='RK45', vectorized=True, 
                                   rtol=1e-5 )
    
    # infodict['message']
    Scl_ODE = YODE2.y[0,:]
    Swb_ODE = YODE2.y[1,:]
    
    
    # Plot results with matplotlib
    plt.figure()
    plt.plot(tOut2, Scl_ODE, 'r-', label='Scl_ODE')
    plt.plot(tOut2, Swb_ODE, 'b-', label='Swb_ODE')
    
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('time (day)')
    plt.ylabel('S (m3/day)')
    plt.title('Evolution of fox and rabbit populations')
    plt.show()
    
    plt.figure()
    plt.plot(Scl_ODE, Swb_ODE, 'b-', label='ODE')
    
    plt.grid()
    plt.legend(loc='best')
    plt.xlabel('Scl')    
    plt.ylabel('Swb')
    plt.title('Evolution of Scl and Swb')
    plt.show()

#%%
#if __name__ == "__main2__":
main2()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    