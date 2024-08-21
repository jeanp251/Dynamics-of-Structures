import math
import numpy as np
import scipy
import matplotlib.pyplot as plt
import scipy.integrate

def read_signal(file_name):
    # Skip header and skip_footer manually obtained
    # Ground Acceleration
    A = np.genfromtxt('./Signals/'+file_name, skip_header=9, skip_footer=239)
    uddg = np.zeros(A.size) # [cm/s^2]
    dt_uddg = 0.02 # Manually obtained
    t_uddg = np.arange(0, (A.size)*dt_uddg, dt_uddg)
    index = 0
    for i in np.nditer(A):
        uddg[index] = i
        index += 1
    
    # Ground Velocity
    B = np.genfromtxt('./Signals/'+file_name, skip_header=346, skip_footer=70)
    udg = np.zeros(B.size) # [cm/s]
    dt_udg = 0.04 # Manually obtained
    t_udg = np.arange(0, (B.size)*dt_udg, dt_udg)
    index = 0
    for i in np.nditer(B):
        udg[index] = i
        index += 1
    
    return t_uddg, uddg, t_udg, udg

def peak_measure(x):
    return max(abs(x.min()), abs(x.max())).round(2)

def get_arias_intensity(t_uddg, uddg):
    def get_quantile_index(vector, quantile):
        index = 0
        for value in vector:
            if value < quantile:
                index += 1
            else:
                break # Exit Loop
        return index
    # CAREFUL WITH UNITS
    square_uddg = np.square(uddg)
    arias_intensity_vector = (1/10)*(np.pi/(2*9810))*scipy.integrate.cumtrapz(square_uddg, t_uddg) # Cumulative trapz
    arias_intensity = (1/10)*(np.pi/(2*9810))*scipy.integrate.trapz(square_uddg, t_uddg) # [cm]
    index_TD5 = get_quantile_index(arias_intensity_vector, 0.05*arias_intensity)
    index_TD95 = get_quantile_index(arias_intensity_vector, 0.95*arias_intensity)

    time_TD5 = t_uddg[index_TD5]
    time_TD95 = t_uddg[index_TD95]

    return arias_intensity_vector, arias_intensity, time_TD5, time_TD95

# INIT POINT
file_name = "IMPVAL1.txt"
t_uddg, uddg, t_udg, udg = read_signal(file_name)

# ARIAS INTENSITY
# CAREFULL WITH THE UNITS!!!
arias_intensity_vector, arias_intensity, time_TD5, time_TD95 = get_arias_intensity(t_uddg, uddg)
print(time_TD5)
print(time_TD95)
IA05 = 0.05*arias_intensity
IA95 = 0.95*arias_intensity
# Cumulative Absolute Displacement CAD
CAD = scipy.integrate.trapz(np.abs(udg), t_udg)
# Strong Motion Duration(TD)
TD = time_TD95 - time_TD5
# Root Mean Square Velocoti Vrms
Vrms = math.sqrt((1/TD)*scipy.integrate.trapz(np.square(udg), t_udg))
print('-'*25)
print('Impact Measures (IMs)')
print('-'*25)
print('PGA [g]: %.2f'%peak_measure(uddg/9810)) # CAREFULL WITH THE UNITS!!!
print('PGV [cm/s]: %.2f'%peak_measure(udg))
print('Ia [cm/s]: %.2f'%arias_intensity)
print('CAD [cm]: %.2f'%CAD)
print('TD [s]: %.2f'%TD)
print('Vrms [cm/s]: %.2f'%Vrms)

# PLOT
#Fig size parameters
a = 10
b = 10

fig1, ax1 = plt.subplots(3,1, figsize = (a,b))
# Ground Acceleration
ax1[0].plot(t_uddg, uddg*(1/9810), color = 'r')
ax1[0].set_xlabel('Time [s]')
ax1[0].set_ylabel('Ground acceleration [g]')
ax1[0].set_xlim([0, max(t_uddg)])

# Ground Velocity
ax1[1].plot(t_udg, udg, color = 'g')
ax1[1].set_xlabel('Time [s]')
ax1[1].set_ylabel('Ground velocity [cm/s]')
ax1[1].set_xlim([0, max(t_udg)])

# Cumulative Integration Ia
ax1[2].plot(np.delete(t_uddg, -1), arias_intensity_vector, color = 'b')
ax1[2].plot([time_TD5, time_TD95], [0.05*arias_intensity, 0.95*arias_intensity], 'ro')
ax1[2].set_xlabel('Time [s]')
ax1[2].set_ylabel('Ia Cummulative [cm]')
ax1[2].set_xlim([0, max(t_udg)])

plt.savefig('./Fig/signal_process_fig1.pdf', format = 'pdf', bbox_inches = 'tight')
plt.show()