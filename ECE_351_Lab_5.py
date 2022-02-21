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
 # Lab 5 #
 #  # 02/22/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as sig

#...........Time............#
steps = 1e-5 # Define step size
t = np.arange(0, 1.2e-3 + steps , steps)



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


#.................Variables......................#
R= 1000
L = 27e-3
C = 100e-9


#...............Hand Calculated impulse response.......#
f = 0.25
def h1(t):
    
    z = 10355.6*np.exp(-5000*t)*np.sin(18584.4*t+1.83363)*step(t)
    return z
           
h1 = h1(t)

#..............scipty signals impulse...................#


num = [0, 1/(R*C),0]
den = [1, 1/(R*C), 1/(C*L)]

tout, yout = sig.impulse((num,den),T=t)



#...............Plots for impulse response...................................#
plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 1)
plt.plot(t,h1)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Hand Calcuted impulse response')


plt.subplot(3, 1, 2)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Amplitude ')
plt.xlabel( 't')
plt.title('Scipy signal impulse')
plt.show

#..............scipty signals step response.........................#

tout, yout = sig.step((num,den),T=t)

plt.figure(figsize = (10, 12))
plt.subplot(3, 1, 1)
plt.plot(tout,yout)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Scipy signal step response')


