# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 13:54:52 2022

@author: learn
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:33:16 2022

@author:
"""

 # ###############################################################
 # #
 # Jayson Haddon #
 # ECE 351 and Section 51 #
 # Lab 7 #
 #  # 03/08/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as sig

#...........Time............#
steps = 1e-2 # Define step size
t = np.arange(0, 10 + steps , steps)



#...........Ramp and Step Function...........#
def ramp(t):
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] > 0):
            y[i]=t[i]
        else: 
            y[i] = 0
    return y
        
def step(t):
    x = np.zeros(t.shape)
    
    for i in range(len(t)):
        if (t[i] < 0):
            x[i]= 0
        else:
            x[i]=1
            
    return x



#....................Defined a(s)...........#
numa = [1,4]
print('A Num =', numa)

dena = sig.convolve([1,1],[1,3])
print('A Den =', dena)

print('Zeros of A = s+4' )
print('Poles of A = (s+1)(s+3)' )

ZA, PA, KA = sig.tf2zpk(numa, dena)

print('sig.tf2zpk method Zeros of A=', ZA)
print('sig.tf2zpk method Pole of A=', PA)



#...................Define g(s)...................#

numg = [1,9]
print('g Num =', numg)

deng = sig.convolve([1,-6,-16],[1,4])
print('g Den =', deng)

print('Zeros of G = s+9' )
print('Poles of G = (s-8)(s+2)(s+4)' )

ZG, PG, KG = sig.tf2zpk(numg, deng)

print('sig.tf2zpk method Zeros of G=', ZG)
print('sig.tf2zpk method Pole of G=', PG)

#................Define B(s).................#

numb = [1,26,168]
print('B Num =', numb)

print('Zeros of B = (s+12)(s+14)' )

ZB = np.roots(numb)

print(' Roots Method Zeros of  B=', ZB)


# No the loop is not stable because there is a positive exponeint.

#.............Define Open Loop Y(s)/X(s)...................#

OLnum = sig.convolve(numa,numg)
OLden = sig.convolve(dena,deng)

tout, yout = sig.step((OLnum,OLden),T=t)

plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('Scipy signal Open Loop')
plt.show


#.............Define Closed Loop Y(s)/X(s)..............#

#H(s) = numa * numg / (dena * numb* numg +  deng * dena)


num1 = sig.convolve(numa, numg)
den1 = sig.convolve(deng, dena) 
den2a = sig.convolve(dena, numb)
den2b = sig.convolve(den2a, numg)
denF = den1 + den2b
print(num1)
print(den1)
print(denF)

tout1, yout1 = sig.step((num1, denF), T=t)

plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 2)
plt.plot(tout1,yout1)
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('Scipy signal impulse Closed loop')
plt.show
