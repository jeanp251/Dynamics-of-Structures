import numpy as np
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
    a3 = (1/(2*β) - 1)*m + dt*(γ/(2*β) - 1)*c
    kt = k + a1

    for i in range(len(p)-1):
        pti_1 = p[i+1] + a1*u[i] + a2*up[i] + a3*upp[i]
        u[i+1] = pti_1/kt
        up[i+1] = (γ/(β*dt))*(u[i+1] - u[i]) + (1-γ/β)*up[i] + dt*(1-γ/(2*β))*upp[i]
        upp[i+1] = (u[i+1] - u[i])/(β*dt**2) - up[i]/(β*dt) - (1/(2*β) - 1)*upp[i]

    return u, up, upp

def maxi(x):
    return max(abs(x.min()), abs(x.max())).round(2)

# Reading the acceleration record
file_name = 'PRQ_19661017.txt'
A = np.genfromtxt('./Signals/'+file_name, skip_header = 37, encoding = 'latin')

dt = A[1,0]

t, ddug_EW, ddug_NS, ddug_UD  = A[:,0], A[:,1], A[:,2], A[:,3]

ddug = ddug_EW

# Structural Parameters
Tn = 0.1
γ, β = 1/2, 1/6

# The method is stable or not
try:
    value = 1/(np.pi*2**.5)*1/(γ-2*β)**.5
except Exception as e:
    print(e)
    value = 10*10

print('Δt/Tn: %.2f'%(dt/Tn))
print('(1/π√2)[1/√(γ-2β)]: %.2f'%value)

if dt/Tn < value:
    print("Newmark's method is stable.")
else:
    print("Try with another values of γ and β")

u, up, upp = newmark(ddug, 0, 0, Tn, 0.05, dt, γ, β)
print('-'*25)
print('Impact Measures (IMs)')
print('PGA EW: %.2f'%maxi(ddug_EW/981))
print('PGA NS: %.2f'%maxi(ddug_NS/981))
print('PGA UD: %.2f'%maxi(ddug_UD/981))

# PLOT
#Fig size parameters
a = 10
b =12

fig1, ax1 = plt.subplots(3,1, figsize = (a,b))
ax1[0].plot(t, ddug_EW*(1/981), color = 'r')
ax1[0].set_xlabel('Time [s]')
ax1[0].set_ylabel('Ground acceleration [g]')
ax1[0].set_xlim([0, max(t)])
ax1[0].set_title('EW')

ax1[1].plot(t, ddug_NS*(1/981), color = 'g')
ax1[1].set_xlabel('Time [s]')
ax1[1].set_ylabel('Ground acceleration [g]')
ax1[1].set_xlim([0, max(t)])
ax1[1].set_title('NS')

ax1[2].plot(t, ddug_UD*(1/981), color = 'b')
ax1[2].set_xlabel('Time [s]')
ax1[2].set_ylabel('Ground acceleration [g]')
ax1[2].set_xlim([0, max(t)])
ax1[2].set_title('UD')
plt.savefig('./Fig/signal_reading_fig1.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()