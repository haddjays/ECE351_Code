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
 # Lab 4 #
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
t = np.arange(-10, 10 + steps , steps)



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
f = 0.25
def h1(t):
    z = np.exp(-2*t) * (step(t)-step(t-3))
    return z
def h2(t):
    x = step(t-2) - step(t-6)
    return x
def h3(t):
   y = np.cos(0.25*2*math.pi*t)*step(t)
   return y
def f(t):
    z = step(t)
    return z

f = f(t)            
h1 = h1(t)
h2 = h2(t)
h3 = h3(t)

#.............Task 1 Plotting functions......#

plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,h1)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h1(t)')



plt.subplot(3, 1, 2)
plt.plot(t,h2)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h2(t)')



plt.subplot(3, 1, 3)
plt.plot(t,h3)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h3(t)')
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

t = np.arange(-10, 10 + steps , steps)

NT = len(t)

textend = np.arange(2*t[0], 2*t[NT-1]+ steps,steps)

fh1 = convo(f,h1)*steps
fh2 = convo(f,h2)*steps
fh3 = convo(f,h3)*steps


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(textend,fh1)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Step Reponse of h1')

plt.subplot(3, 1, 2)
plt.plot(textend,fh2)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Step Reponse of h2')

plt.subplot(3, 1, 3)
plt.plot(textend,fh3)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Step Reponse of h3')
plt.show

#.................Part 3..........#
t = np.arange (-20, 20 + steps, steps)

def h1c(t):
    z = 1/2*(1 - np.exp(-2*t))*step(t) - 1/2*(1 - np.exp(-2*(t-3)))*step(t-3)
    return z
def h2c(t):
    z = ((t-2)*step(t-2))-((t-6)*step(t-6))
    return z
def h3c(t):
    w = 0.25*2*math.pi
    z = 1/w*np.sin(w*t)*step(t)
    return z

h1c = h1c(t)
h2c = h2c(t)
h3c = h3c(t)


plt.figure(figsize = (10, 7))
plt.subplot(3, 1, 1)
plt.plot(t,h1c)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h1c(t)')

plt.subplot(3, 1, 2)
plt.plot(t,h2c)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h2(t)')

plt.subplot(3, 1, 3)
plt.plot(t,h3c)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('h3(t)')
plt.show()
