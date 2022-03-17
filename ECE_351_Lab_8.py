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
 # Lab 8 #
 #  # 03/15/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as sig

#...........Time............#
steps = 1e-2 # Define step size
t = np.arange(0, 20 + steps , steps)



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



#def a0(k):
#    x = 0
#    return x


def ak(k):
    x = 0
    return x


def bk(k):
    x = (2/(k*math.pi))*(1-np.cos(k*math.pi))
    return x

print('Ak of 0 =', ak(0) )
print('Ak of 1 =', ak(1) )
print('Ab of 1 =', bk(1) )
print('Ab of 2 =', bk(2) )
print('Ab of 3 =', bk(3) )



def x(N):
    T = 8
    w0 = (2*math.pi)/T
    aksum = 0
    bksum = 0
    for num in np.arange(1,N+1):
        num = ak(num)*np.cos(num*w0*t)
        aksum = aksum + num
        #print('AKSum =', aksum)
    for num in np.arange(1,N+1):
        num = bk(num)*np.sin(num*w0*t)
        bksum = bksum + num
        #print('BKSum =', bksum)
    x = 1/2*0 + aksum+bksum
    return x

#print('bk =',x(t,5) )

plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 1)
plt.plot(t,x(1))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 1')

plt.subplot(3, 1, 2)
plt.plot(t,x(3))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 3')

plt.subplot(3, 1, 3)
plt.plot(t,x(15))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 15')

plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 1)
plt.plot(t,x(50))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 50')

plt.subplot(3, 1, 2)
plt.plot(t,x(150))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 150')

plt.subplot(3, 1, 3)
plt.plot(t,x(1500))
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('N = 1500')
plt.show


    
