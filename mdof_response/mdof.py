import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

def newmark(ug, u0=0, up0=0, Tn=0.1, ξ=0.05, Δt=0.01, γ=1/2, β=1/6):
    m = 1. # Arbitrary mass
    p = -m*ug
    
    k = m*(2*np.pi/Tn)**2
    ωn = 2*np.pi/Tn
    c = 2*ξ*m*ωn

    u = np.zeros(len(p))
    u[0] = u0
    up = np.zeros(len(p))
    up[0] = up0
    upp = np.zeros(len(p))
    upp[0] = (p[0] - c*up0 - k*u0)/m

    a1 = m/(β*Δt**2) + γ*c/(β*Δt)
    a2 = m/(β*Δt) + (γ/β - 1)*c
    a3 = (1/(2*β) - 1)*m + Δt*(γ/(2*β) - 1)*c
    kt = k + a1

    for i in range(len(p)-1):
        pti_1 = p[i+1] + a1*u[i] + a2*up[i] + a3*upp[i]
        u[i+1] = pti_1/kt
        up[i+1] = (γ/(β*Δt))*(u[i+1] - u[i]) + (1-γ/β)*up[i] + Δt*(1-γ/(2*β))*upp[i]
        upp[i+1] = (u[i+1] - u[i])/(β*Δt**2) - up[i]/(β*Δt) - (1/(2*β) - 1)*upp[i]

    return u, up, upp

def tridiag(k=2.1, n=5):
    # Function to create a shear frame Stiffness Matrix [K]
    # With the same inter-story stiffness  "ki" [N/m]
    aa = [-k for i in range(n-1)]
    bb = [2*k for i in range(n)]
    bb[-1] = k

    return np.diag(aa, -1) + np.diag(bb, 0) + np.diag(aa, 1)

# Creating the Stiffness and Mass Matrixes
n = 9
K = tridiag(5000.0, n)
m = 1 # Story Mass [kg]
M = np.identity(n)*m
print('- Stiffness Matrix [K]')
print(K)
print('-'*100)
print('- Mass Matrix [M]')
print(M)
print('-'*100)

# Obtaining the eigenvalues and eigenvectors
Ω, Φ = linalg.eigh(K,M)
print('- Eigenvectors [Φ]')
print(Φ)
print('-'*100)
print('- Eigenvalues [Ω] wi^2')
print(Ω)

# Checking that the modes are normalized with the mass
for i in range(n):
    print(Φ[:,i].T@M@Φ[:,i])

# Calculating the Periods and Mass Participation Factor
PF, Tn = [], []

print('-'*100)
print('Natural Periods Tn and Participation Factors (PF)')
for i in range(n):
    PF.append(sum(Φ[:,i].T@M))
    Tn.append(2.0*np.pi/Ω[i]**0.5)
    print('T[%i]: %5.3f, PF[%i]: %7.3f'%(i+1, Tn[i], i+1, PF[i]))

print('-'*100)
print('- Sum of the participation mass factors:')
print(sum(PF))
print('-'*100)

# READING THE SIGNAL AND ANALYSIS PARAMETERS
file_name = 'PRQ_19661017.txt'
A = np.genfromtxt('./Signals/'+file_name, skip_header = 37, encoding = 'latin')
dt = A[1,0]

t, ddug = A[:,0], A[:,1]
γ, β = 1/2, 1/6
N = len(ddug)

response = np.zeros((N,n))

for i in range(n):
    # n: degrees of freedom
    # N: number of points of the seismic record
    x, dx, ddx = newmark(-PF[i]*ddug, u0 = 0, up0 = 0, Tn = Tn[i], ξ=0.05, Δt=dt, γ = 1/2, β=1/6)

    for j in range(N):
        response[j] = x[j]*Φ[:,i] + response[j]

# PLOTTING
a = 20
b = 5

f1 = plt.figure(figsize = (a,b))
plt.plot(t, response)
plt.legend(loc = 'best', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sa [cm/s2]', size = 16)

plt.show()