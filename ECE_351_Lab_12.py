# -*- coding: utf-8 -*-
"""
Created on Tue Apr 12 13:44:47 2022

@author: learn
"""

 # ###############################################################
 # #
 # Jayson Haddon #
 # ECE 351 and Section 51 #
 # Lab 12 #
 #  # 04/26/2022
 # Any other necessary information needed to navigate the file #
 # #
 # ##############################################
 
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.signal as sig
import control as con
import pandas as pd
import scipy.fftpack

# load input signal
df = pd. read_csv ('NoisySignal.csv ')

t = df['0']. values
sensor_sig = df['1']. values

plt. figure ( figsize = (10 , 7))
plt. plot (t, sensor_sig )
plt. grid ()
plt. title ('Noisy Input Signal ')
plt. xlabel ('Time [s]')
plt. ylabel ('Amplitude [V]')
plt. show ()


##### fast graphing
def make_stem(ax ,x,y, color='k', style='solid',label='',linewidths =2.5 ,** kwargs):
    ax.axhline(x[0],x[-1],0, color='r')
    ax.vlines(x, 0 ,y, color=color , linestyles=style , label=label , linewidths=linewidths)
    ax.set_ylim ([1.05*y.min(), 1.05*y.max()])



# Fast Fourier
def cleanfastfour(x,fs):
    N = len(x)
    X_fft = scipy.fftpack.fft(x)
    X_fft_shifted = scipy.fftpack.fftshift(X_fft) 

    freq = np.arange(-N/2, N/2)*fs/N
    X_mag = np.abs(X_fft_shifted)/N 
    X_phi = np.angle(X_fft_shifted)

    
    for i in range(len(X_mag)):
     if abs(X_mag[i]) < 1e-10:
         X_phi[i] = 0
    return freq, X_mag, X_phi



# Finding the different magnitudes for different frequencies

fs = 1e6 #Will cover the entire range of the signal

freq, X_mag, X_phi = cleanfastfour(sensor_sig,fs)

#Magnitude of Entire Range
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Unfiltered Range of  to 10000 Hz')
plt.xlim(0,100000)
plt.grid ()
plt.show ()

#Magnitude of range from 0 to 200 Hz

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Unfiltered Range of 0 to 200 Hz')
plt.xlim(0,200)
plt.grid ()
plt.show ()


#Magnitude of range up to 1800 Hz

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Unfiltered Range of 0 to 1800 Hz')
plt.xlim(0,1800)
plt.grid ()
plt.show ()

#Magnitude range of 1800 to 2000

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Unfiltered Range of 1800 to 2000 Hz')
plt.xlim(1800,2000)
plt.grid ()
plt.show ()

#Magnitude range of 2000 to 10000

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Unfiltered Range of 2000 to 10000 Hz')
plt.xlim(2000,10000)
plt.grid ()
plt.show ()

#-------------Part 2-------------#
#c1 = 4.42097e-9
#c2 = 3.97887e-9
#r1 = 20000
#r2 = 20000

#Use bandpass filter from lab 10
#calculate L with center freqency of 1900, R = 10000, and C = 100e-9
#1900 = 1/(2*pi*sqrt(L*100e-9))


R = 3700
C = 8.84194e-9
L = 0.795775
#C = 9.5e-9
#L = 1
#C = 9e-9
num = [C*R,0]
den = [C*L,C*R,1]
steps = 1
f = np.arange(1, 1e6 + steps, steps)
w = f*2*np.pi

zfunction, pfunction = sig.bilinear(num, den, fs)
yt = sig.lfilter(zfunction, pfunction, sensor_sig)




plt.figure(figsize = (10,7))
plt.plot(t, yt)
plt.grid()
plt.title('Filtered NoisySignal')
plt.xlabel('t')
plt.ylabel('Magnitude')
plt.show()

#Bode Plot of entire range
sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

steps = 1
f = np.arange(1, 2000 + steps, steps)
w = f*2*np.pi

sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

steps = 1
f = np.arange(1800, 2000 + steps, steps)
w = f*2*np.pi

sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

steps = 1
f = np.arange(2000, 10000 + steps, steps)
w = f*2*np.pi

sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

steps = 1
f = np.arange(10000, 100000 + steps, steps)
w = f*2*np.pi

sys = con.TransferFunction ( num , den )
plt.figure ( figsize = (10 , 7) )
_ = con.bode ( sys , w , dB = True , Hz = True , deg = True , Plot = True )

freq, X_mag, X_phi = cleanfastfour(yt,fs)

#Magnitude of Entire Range
fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Entire Range')
plt.xlim(0,100000)
plt.grid ()
plt.show ()

#Magnitude of range from 0 to 200 Hz

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Range of 0 to 200 Hz')
plt.xlim(0,200)
plt.grid ()
plt.show ()


#Magnitude of range up to 1800 Hz

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Range of 0 to 1800 Hz')
plt.xlim(0,1800)
plt.grid ()
plt.show ()

#Magnitude range of 1800 to 2000

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Range of 1800 to 2000 Hz')
plt.xlim(1800,2000)
plt.grid ()
plt.show ()

#Magnitude range of 2000 to 10000

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Range of 2000 to 10000 Hz')
plt.xlim(2000,10000)
plt.grid ()
plt.show ()

#Magnitude range of 10000 to 100000

fig , ax = plt . subplots ( figsize =(10 , 7) )
make_stem ( ax , freq , X_mag )
plt.ylabel ('|X(f)|')
plt.xlabel ('f[Hz]')
plt.title('Filtered Range of 10000 to 100000 Hz')
plt.xlim(10000,100000)
plt.grid ()
plt.show ()
