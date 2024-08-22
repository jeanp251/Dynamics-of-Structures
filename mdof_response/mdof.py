import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt
from matplotlib import animation, rc

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
# Sd vs Time
f1 = plt.figure(figsize = (a,b))

for i in range(n):
    plt.plot(t, response[:,i], label = 'DOF %i' %(i+1))

plt.legend(loc = 'upper right', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sd [cm]', size = 16)
plt.axis([0, t[-1], -3, 3])
plt.show()

rc('animation', html = 'jshtml')

# First set the figure, the axis, and the plot we want to animate
fig = plt.figure(figsize=(10,10))
ax = plt.axes(xlim = (-2.5, 2.5), ylim = (0, n))
line, = ax.plot([], [], 'ko:', lw=2, label = '0.00s')

# Initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    return line,

# Animation function: This function is called sequentially
def animate(i):
    x = np.insert(response[i],0,0)
    y = np.arange(0, n+1, 1)

    line.set_data(x,y)
    line.set_label('%6.2fs'%(i*0.02))
    ax.legend()

    return line,

# Call the Animator, blit = True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func = init, frames = N, interval = 50, blit = True)

# Save the animation as mp4 This requires ffmpeg or mencoder to be 
#anim.save('basic_animation.mp4', fps = 30, extra_args = ['-vcodec', 'libx264'])
plt.show()