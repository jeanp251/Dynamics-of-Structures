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
    a3 = (1/(2*β) - 1)*m + Δt*(γ/(2*β) - 1)*c
    kt = k + a1

    for i in range(len(p)-1):
        pti_1 = p[i+1] + a1*u[i] + a2*up[i] + a3*upp[i]
        u[i+1] = pti_1/kt
        up[i+1] = (γ/(β*Δt))*(u[i+1] - u[i]) + (1-γ/β)*up[i] + Δt*(1-γ/(2*β))*upp[i]
        upp[i+1] = (u[i+1] - u[i])/(β*Δt**2) - up[i]/(β*Δt) - (1/(2*β) - 1)*upp[i]

    return u, up, upp

def maxi(x):
    return max(abs(x.min()), abs(x.max())).round(2)

# Reading the acceleration record
file_name = 'PRQ_19661017.txt'
A = np.genfromtxt('./Signals/'+file_name, skip_header = 37, encoding = 'latin')

Δt = A[1,0]

t, ug = A[:,0], A[:,1]

# Structural Parameters
Tn = 0.25
γ, β = 1/2, 1/6

# The method is stable or not
try:
    value = 1/(np.pi*2**.5)*1/(γ-2*β)**.5
except Exception as e:
    print(e)
    value = 10*10

relation = Δt/Tn
print('Δt/Tn: %.2f'%relation)
print('(1/π√2)[1/√(γ-2β)]: %.2f'%value)

if Δt/Tn < value:
    print("Newmark's method is stable.")
else:
    print("Try with another values of γ and β")

u, up, upp = newmark(ug, 0, 0, Tn, 0.05, Δt, γ, β)

# PLOT
a = 20
b = 5

f1 = plt.figure(figsize = (a,b))
plt.plot(t, upp, 'r', linewidth = 0.5, alpha = 1, label = 'upp max value = '+str(maxi(upp)) + 'cm/s2')
plt.title('Elastic response of a 1dof with Tn = '+str(Tn), size = 16)
plt.legend(loc='best', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sa [cm/s2]', size = 16)
plt.savefig('./Fig/Image1.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

f2 = plt.figure(figsize = (a,b))
plt.plot(t, ug, 'g', linewidth = 0.5, alpha = 1, label = 'ug max value = '+str(maxi(ug)) + 'cm/s2')
plt.title('Elastic response of a 1dof with Tn = '+ str(Tn), size = 16)
plt.legend(loc='best', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sa [cm/s2]', size = 16)
plt.savefig('./Fig/Image2.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

f3 = plt.figure(figsize = (a,b))
plt.plot(t, ug + upp, 'b', linewidth = 0.5, alpha = 1, label = 'ug+upp max value = '+str(maxi(ug+upp)) + 'cm/s2')
plt.title('Elastic absolute response of a 1dof with Tn = '+ str(Tn), size = 16)
plt.legend(loc='best', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sa [cm/s2]', size = 16)
plt.savefig('./Fig/Image3.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

f4 = plt.figure(figsize = (a,b))
plt.plot(t, u, 'lime', linewidth = 0.5, alpha = 1, label = 'u max value = '+str(maxi(u)) + 'cm/s2')
plt.title('Elastic response of a 1dof with Tn = '+ str(Tn), size = 16)
plt.legend(loc='best', frameon = True, fontsize = 'x-large')
plt.xlabel('Time [s]', size = 16)
plt.ylabel('Sd [cm/s2]', size = 16)
plt.savefig('./Fig/Image4.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()

# RESPONSE SPECTRUM

T_arange = np.arange(0,3,0.05)
n = len(T_arange)
resp_arange = np.zeros(n)

i = 0
for Tn in T_arange:
    if Tn == 0:
        resp_arange[i] = maxi(ug)
        i = i+1
        continue
    u, up, upp = newmark(ug, 0, 0, Tn, 0.05, Δt, γ=1/2, β=1/6)
    resp_arange[i] = maxi(upp + ug)
    i = i+1

a, b = 10, 10
f5 = plt.figure(figsize=(a,b))
plt.plot(T_arange, resp_arange, label = "Response Spectrum")
plt.title('Pseudo-acceleration response spectrum', size = 16)
plt.legend(loc = 'best', frameon = True, fontsize = 'x-large')
plt.xlabel('T [s]', size = 16)
plt.ylabel('Sa [gal]', size = 16)
plt.savefig('./Fig/Image5.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()