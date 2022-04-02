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
 # Lab 10 #
 #  # 03/15/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as sig
import control as con

#...........Time............#
steps = 1e-2 # Define step size
t = np.arange(0, 0.01 + steps , steps)

#..........W.................#
steps = 1e3 # Define step size
w = np.arange(1e3, 1e6 + steps , steps)

R = 1000
L = 27e-3
C = 100e-9

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

def adjustHdeg(Hdeg):
    for i in range(len(Hdeg)):
        if Hdeg[i] > 90:
            Hdeg[i] = Hdeg[i] - 180
    return Hdeg

R = 1e3
L = 27e-3
C = 100e-9


maghs = (20*np.log10((w/(R*C))/(np.sqrt(w**4 + (1/(R*C)**2 - 2/(L*C))*w**2 + (1/(L*C))**2))))

Hdeg = (np.pi/2 - np.arctan((1/(R*C)*w)/(-w**2 + 1/(L*C)))) * 180/np.pi
Hdeg = adjustHdeg(Hdeg)

plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w, maghs)
plt.grid()
plt.ylabel('Magnitude in dB')
plt.title('Task 1')

plt.subplot(2,1,2)
plt.semilogx(w, Hdeg)
plt.yticks([-90, -45, 0, 45, 90])
plt.ylim([-90,90])
plt.grid()
plt.ylabel('Phase in degrees')
plt.xlabel('Frequency in rad/s')
plt.show()


#.............Part 2.................#

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]


w, maghs , Hdeg = sig.bode((num, den), w)


plt.figure(figsize = (10,7))
plt.subplot(2,1,1)
plt.semilogx(w, maghs)
plt.grid()
plt.xlim([1e3, 1e6])
plt.title('Task 2')

plt.subplot(2,1,2)
plt.semilogx(w, Hdeg)
plt.grid()
plt.xlim([1e3, 1e6])
plt.ylim([-90,90])
plt.xlabel('Frequency in rad/s')
plt.show()


#...............Part 3..................#

num = [1/(R*C), 0]
den = [1, 1/(R*C), 1/(L*C)]
sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )
# use _ = ... to suppress the output

#.............Part 4..................#
fs = 50000*2*np.pi
steps = 1/fs
t = np.arange(0, 0.01 + steps , steps)



def function(t):
    x = np.cos(2*np.pi*100*t)+np.cos(2*np.pi*3042*t)+np.sin(2*np.pi*50000*t)
    return x
function = function(t)

plt.figure ( figsize = (10 , 7) )
plt.subplot (1 , 1 , 1)
plt.plot(t, function)
plt.grid()
plt.title('Task 4')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()



zfunction, pfunction = sig.bilinear(num, den, fs)


yt = sig.lfilter(zfunction, pfunction, function)



plt.figure(figsize = (10,7))
plt.plot(t, yt)
plt.grid()
plt.title('Task 5')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()
