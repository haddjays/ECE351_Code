# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 13:36:21 2022

@author: learn
"""

 # ###############################################################
 # #
 # Jayson Haddon #
 # ECE 351 and Section 51 #
 # Lab 2 #
 #  # 02/1/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################
 
import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t = np.arange(-5, 10 + steps , steps) # Add a step size to make sure the
# plot includes 5.0. Since np.arange () only
# goes up to , but doesn’t include the
# value of the second argument
print('Number of elements: len(t) = ', len(t), '\nFirst Element: t[0] = ', t[0], 
      ' \nLast Element: t[len(t) - 1] = ', t[len(t) - 1])
# Notice the array might be a different size than expected since Python starts
 # at 0. Then we will use our knowledge of indexing to have Python print the
# first and last index of the array. Notice the array goes from 0 to len() - 1

# --- User - Defined Function ---
# Create output y(t) using a for loop and if/else statements
def func1(t): # The only variable sent to the function is t
    y = np.zeros(t.shape)
    
    for i in range(len(t)):
            y[i] = np.cos(t[i]) #create a cosine graph
    return y

#y = func1(t) # call the function we just created

#plt.figure(figsize = (10, 7))
#plt.subplot(2, 1, 1)
#plt.plot(t, y)
#plt.grid()
#plt.ylabel('Amplitude')
#plt.xlabel( 't')
#plt.title('Part 4 Cosine Graph ')


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

def deriv(t):
    u = np.zeros(t.shape)
    
    u = ramp(t) - ramp((t-3)) +5*step((t-3))-2*step((t-6))-2*ramp((t-6))
    return u
u = deriv(t)

y = ramp(t) # Ramp function
x = step(t) # Step function
z = ramp(t) - ramp((t-3)) +5*step((t-3))-2*step((t-6))-2*ramp((t-6))

dt = np.diff(t)
dy = np.diff((deriv(t)))/dt

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, u)
plt.plot(t[range(len(dy))],dy)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Part 6 ')
plt.show
