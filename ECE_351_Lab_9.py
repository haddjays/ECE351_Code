# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 13:33:32 2022

@author: learn
"""
 # ###############################################################
 # #
 # Jayson Haddon #
 # ECE 351 and Section 51 #
 # Lab 9#
 #  # 03/29/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ###############################################################


import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal
import scipy.fftpack






fs = 1000


#...........Time............#
steps = 1/fs
t = np.arange(0, 2 , steps)

x = np.cos(2*t*math.pi)

def fastfour(x,fs):
    N = len(x) 
    X_fft = scipy.fftpack.fft(x) 
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 

    freq = np.arange(-N/2, N/2)*fs/N 
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted) 

    plt.stem(freq , X_mag) 
    plt.stem(freq , X_phi) 
    return freq, X_mag, X_phi

#----------------Task 1---------------------#
freq, X_mag, X_phi = fastfour(x,fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 1: cos(2*pi*t)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#----------------Task 2---------------------#
x = 5*np.sin(2*math.pi*t)

freq, X_mag, X_phi = fastfour(x,fs)


plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 2: 5sin(2pit)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#----------------Task 3---------------------#
x = 2*np.cos((2*math.pi*2*t)-2)+np.sin((2*math.pi*6*t)+3)**2

freq, X_mag, X_phi = fastfour(x,fs)


plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 3: 2cos((2pi*2t)-2)+sin^2((2pi*6t)+3)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-15, 15)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-15, 15)
plt.xlabel ('f[Hz]')
plt.show ()

#----------------Task 4---------------------#

def cleanfastfour(x,fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 

    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted)

    plt.stem(freq , X_mag)
    plt.stem(freq , X_phi)
    
    for i in range(len(X_mag)):
     if abs(X_mag[i]) < 1e-10:
         X_phi[i] = 0
    return freq, X_mag, X_phi

x = np.cos(2*t*math.pi)

freq, X_mag, X_phi = cleanfastfour(x,fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4a: cos(2*pi*t)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#--------------------------Task 4b----------------------#

x = 5*np.sin(2*math.pi*t)

freq, X_mag, X_phi = cleanfastfour(x,fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4b: 5sin(2pit)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

#--------------------------Task 4c----------------------#

x = 2*np.cos((2*math.pi*2*t)-2)+np.sin((2*math.pi*6*t)+3)**2

freq, X_mag, X_phi = cleanfastfour(x,fs)

plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 4c: 2cos((2pi*2t)-2)+sin^2((2pi*6t)+3)')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()

def ak(k):
    x = 0
    return x


def bk(k):
    x = (2/(k*math.pi))*(1-np.cos(k*math.pi))
    return x

def square(N):
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

steps = 1/fs
t = np.arange(0, 16 , steps)
x = square(15)

freq, X_mag, X_phi = cleanfastfour(x,fs)


plt.figure ( figsize = (10 , 7) )
plt.subplot (3 , 1 , 1)
plt.plot (t , x)
plt.grid ()
plt.ylabel ('k = 1')
plt.xlabel ('t')
plt.title('Task 5: Square')

plt.subplot (3 , 2 , 3)
plt.stem(freq, X_mag)
plt.grid ()
plt.ylabel ('|H(f)|')

plt.subplot (3 , 2 , 5)
plt.stem (freq, X_phi)
plt.grid ()
plt.ylabel ('/_X(f)')
plt.xlabel ('f[Hz]')

plt.subplot (3 , 2 , 4)
plt.stem(freq, X_mag)
plt.grid ()
plt.xlim(-2, 2)

plt.subplot (3 , 2 , 6)
plt.stem (freq, X_phi)
plt.grid ()
plt.xlim(-2, 2)
plt.xlabel ('f[Hz]')
plt.show ()
