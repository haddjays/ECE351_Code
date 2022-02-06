# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 13:33:16 2022

@author:
"""

 # ###############################################################
 # #
 # Jayson Haddon #
 # ECE 351 and Section 51 #
 # Lab 3 #
 #  # 02/8/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal

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
        if (t[i] > 0):
            x[i]= 1
        else:
            x[i]=0
            
    return x

#...............functions to graph.......#

def f1(t):
    z = step(t-2) - step(t-9)
    
    return z
def f2(t):
    x = np.exp(-t) * step(t)
    return x
def f3(t):
    y = ramp(t-2)*(step(t-2)-step(t-3))+ramp(4-t)*(step(t-3)-step(t-4))
    return y

            
f1 = f1(t)
f2 = f2(t)
f3 = f3(t)

#.............Task 1 Plotting functions......#

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,f1)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f1(t)')



plt.subplot(3, 1, 2)
plt.plot(t,f2)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f2(t)')



plt.subplot(3, 1, 3)
plt.plot(t,f3)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f3(t)')
plt.show()


#.............Convolution............#

def convo(f1, f2):
    Nf1 = len(f1)
    Nf2 = len(f2)
    f1extend = np.append(f1, np.zeros((1,Nf2-1)))
    f2extend = np.append(f2, np.zeros((1,Nf1-1)))
    result = np.zeros(f1extend.shape)
    for i in range (Nf2 + Nf1-2):
        result[i]=0
        for j in range (Nf1):
                if(i-j+1 > 0):
                    try:
                        result[i]+= f1extend[j]*f2extend[i-j+1]
                    except:
                        print(i,j)
    return result

#................Adjusting Time............#

t = np.arange(0, 20 + steps , steps)

NT = len(t)

textend = np.arange(0, 2*t[NT-1],steps)


#..................Set up Convolution functions............#
f12 = convo(f1,f2)*steps
f23 = convo(f2,f3)*steps
f13 = convo(f1,f3)*steps


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(textend,f12)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f12')

plt.subplot(3, 1, 2)
plt.plot(textend,f23)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f23')

plt.subplot(3, 1, 3)
plt.plot(textend,f13)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f13')
plt.show

#................Check..............#

cf12 = scipy.signal.convolve(f1,f2)
cf23 = scipy.signal.convolve(f2,f3)
cf13 = scipy.signal.convolve(f1,f3)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(textend,f12)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f12')

plt.subplot(3, 1, 2)
plt.plot(textend,cf12)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('cf12')

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(textend,f23)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f23')

plt.subplot(3, 1, 2)
plt.plot(textend,cf23)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('cf23')

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(textend,f13)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('f13')

plt.subplot(3, 1, 2)
plt.plot(textend,cf13)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('cf13')
