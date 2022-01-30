import numpy as np
import matplotlib.pyplot as plt

#plt.rcParams.update({'fontsize': 14}) # Set font size in plots

steps = 1e-2 # Define step size
t = np.arange(0, 10 + steps , steps) # Add a step size to make sure the
# plot includes 5.0. Since np.arange () only
# goes up to , but doesnt include the
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

y = func1(t) # call the function we just created
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
            y[i]=0
    return x

y = ramp(t) # Ramp function
x = step(t) # Step function

z = ramp(t) - ramp(t-3) +5*step(t-3)-2*step(t-6)-2*ramp(t-6)

plt.figure(figsize = (10, 7))
plt.subplot(2, 1, 1)
plt.plot(t, z)
plt.grid()
plt.ylabel('Amplitude')
plt.xlabel( 't')
plt.title('Part 4 Cosine Graph ')
