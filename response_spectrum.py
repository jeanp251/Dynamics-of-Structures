import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import pandas as pd

def state_space(xi, k, omega_n, dt, uddg):
    PGD = np.zeros(len(omega_n))

    for i in range(len(omega_n)):
        wn = omega_n[i]
        A = np.array([[0, 1], [-wn**2, -2*xi*wn]])
        D, V = np.linalg.eig(A)

        ep = np.array([[np.exp(D[0]*dt), 0], [0, np.exp(D[1]*dt)]])
        Ad = V.dot(ep).dot(np.linalg.inv(V))
        Bd = np.linalg.inv(A).dot(Ad - np.eye(len(A)))

        z = np.array([[0], [0]])
        u = np.zeros(len(uddg))
        udot = np.zeros(len(uddg))

        for j in range(len(uddg)):
            F = np.array([[0], [-uddg[j]*9.81]])
            z = np.real(Ad).dot(z) + np.real(Bd).dot(F)
            u[j] = z[0,0]
            udot[j] = z[1,0]
        
        PGD[i] = max(abs(u))
        
    return PGD


# EL-CENTRO READ SIGNAL
signal = pd.read_excel('./Signals/ELC.xls')
t = signal.iloc[:,0] # t [s]
uddg = signal.iloc[:,1] # Ground Acceleration [g]
plt.plot(t, uddg)
plt.show()

# INPUT
xi = 0.02
m = 1 # [kg]
dt = t[1] - t[0]
T = np.linspace(0.01, 3, 350) # Array of natural periods [s]
omega_n = 2*np.pi/T
k = m*omega_n**2

# SOLUTION 
D = state_space(xi, k, omega_n, dt, uddg)

# RESPONSE SPECTRUM
V = omega_n*D
A = omega_n*V/9.81

# PLOT RESPONSE SPECTRUM
a = 8
b = 8
f2 = plt.figure(figsize = (a,b))
plt.subplot(3,1,1)
plt.plot(T, D, 'r', linewidth = 2)
plt.title('RESPONSE SPECTRUM', fontsize = 12, fontweight = 'bold')
plt.ylabel('Deformation \nResponse Spectrum\n D [m]', fontsize = 10, fontweight = 'bold')
plt.xlim(0, 3)
plt.ylim(0, 0.5)

plt.subplot(3,1,2)
plt.plot(T, V, 'r', linewidth = 2)
plt.ylabel('Pseudo-Velocity \nResponse Spectrum\n V [m/s]', fontsize = 10, fontweight = 'bold')
plt.xlim(0, 3)
plt.ylim(0, 1.5)

plt.subplot(3,1,3)
plt.plot(T, A, 'r', linewidth = 2)
plt.ylabel('Pseudo-Acceleration \nResponse Spectrum\n A [g] [m/s]', fontsize = 10, fontweight = 'bold')
plt.xlim(0, 3)
plt.ylim(0, 1.5)
plt.savefig('./Fig/response_spectrum_fig1.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()