import numpy as np
from random import random
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# The function that generates the signals is imported
from seniales_sep import signal_generation  

data = signal_generation()  # The signs are contained in a dictionary
voltage_1, current_1 = data["Node 1"]
voltage_2, current_2 = data["Node 2"]
voltage_3, current_3 = data["Node 3"]

#NODE 1
fig0, ax0 = plt.subplots(1, 2, figsize=(16, 3))
fig0.suptitle('NODE 1')
ax0[0].plot(voltage_1.T)  
ax0[0].set_title("Voltages")
ax0[0].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax0[1].plot(current_1.T)  
ax0[1].set_title("Currents")
ax0[1].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
#NODE 2
fig1, ax1 = plt.subplots(1, 2, figsize=(16, 3))
fig1.suptitle('NODE 2')
ax1[0].plot(voltage_2.T)  
ax1[0].set_title("Voltages")
ax1[0].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax1[1].plot(current_2.T)  
ax1[1].set_title("Currents")
ax1[1].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
#NODE 3
fig2, ax2 = plt.subplots(1, 2, figsize=(16, 3))
fig2.suptitle('NODE 3')
ax2[0].plot(voltage_3.T)  
ax2[0].set_title("Voltages")
ax2[0].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax2[1].plot(current_3.T)  
ax2[1].set_title("Currents")
ax2[1].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
plt.show()

# Calculation of instantaneous power
P_ph1 = voltage_1*current_1 #Node 1
P_ph2 = voltage_2*current_2 #Node 2
P_ph3 = voltage_3*current_3 #Node 3
#Graphics of the INSTANT power. 
fig1, ax1 = plt.subplots(3, 1, figsize=(12, 6))
fig1.suptitle('INSTANT POWER PER PHASE')
# Instant power per phase node 1
ax1[0].plot(P_ph1.T)
ax1[0].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax1[0].set_title("Node 1")
ax1[0].grid()
# Instant power per phase node 2
ax1[1].plot(P_ph2.T)
ax1[1].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax1[1].set_title("Node 2")
ax1[1].grid()
# Instant power per phase node 3
ax1[2].plot(P_ph3.T)
ax1[2].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax1[2].set_title("Node 3")
ax1[2].grid()
plt.xlabel('TIME DOMAIN')
plt.show()

#The signal was sampled for 500 data. 
N = 500
C = 1/N
#CALCULATION OF THE VOLTAGES
V_N1 = C*np.sum(voltage_1**2, axis=1)
V_N2 = C*np.sum(voltage_2**2, axis=1)
V_N3 = C*np.sum(voltage_3**2, axis=1)
#CALCULATION OF THE CURRENTS. 
I_L1 = C*np.sum(current_1**2,axis=1)
I_L2 = C*np.sum(current_2**2,axis=1)
I_L3 = C*np.sum(current_3**2,axis=1)
print("VOLTAGES [V]")
print(f'NODE 1: {V_N1[0]: .4f} {V_N1[1]: .4f} {V_N1[2]: .4f}')
print(f'NODE 2: {V_N2[0]: .4f} {V_N2[1]: .4f} {V_N2[2]: .4f}')
print(f'NODE 3: {V_N3[0]: .4f} {V_N3[1]: .4f} {V_N3[2]: .4f}')
print("CURRENTS [A]")
print(f'NODE 1: {I_L1[0]: .4f} {I_L1[1]: .4f} {I_L1[2]: .4f}')
print(f'NODE 2: {I_L2[0]: .4f} {I_L2[1]: .4f} {I_L2[2]: .4f}')
print(f'NODE 3: {I_L3[0]: .4f} {I_L3[1]: .4f} {I_L3[2]: .4f}')

#Voltages RMS Values. 
RMS_V_N1 = np.sqrt(V_N1)
RMS_V_N2 = np.sqrt(V_N2)
RMS_V_N3 = np.sqrt(V_N3)
#Currents RMS Values. 
RMS_I_L1 = np.sqrt(I_L1)
RMS_I_L2 = np.sqrt(I_L2)
RMS_I_L3 = np.sqrt(I_L3)
print("VOLTAGES (RMS) [V]")
print(f'NODE 1: {RMS_V_N1[0]: .4f} {RMS_V_N1[1]: .4f} {RMS_V_N1[2]: .4f}')
print(f'NODE 2: {RMS_V_N2[0]: .4f} {RMS_V_N2[1]: .4f} {RMS_V_N2[2]: .4f}')      
print(f'NODE 3: {RMS_V_N3[0]: .4f} {RMS_V_N3[1]: .4f} {RMS_V_N3[2]: .4f}')
print("CURRENTS (RMS) [A]")
print(f'NODE 1: {RMS_I_L1[0]: .4f} {RMS_I_L1[1]: .4f} {RMS_I_L1[2]: .4f}')
print(f'NODE 2: {RMS_I_L2[0]: .4f} {RMS_I_L2[1]: .4f} {RMS_I_L2[2]: .4f}')      
print(f'NODE 3: {RMS_I_L3[0]: .4f} {RMS_I_L3[1]: .4f} {RMS_I_L3[2]: .4f}')

#CALCULATION OF THE APPARENT POWER PER PHASE. 
S_N1 = RMS_V_N1 * RMS_I_L1
S_N2 = RMS_V_N2 * RMS_I_L2
S_N3 = RMS_V_N3 * RMS_I_L3
print("APPARENTE POWER [VA]")
print(f'NODO 1: {S_N1[0]: .4f} {S_N1[1]: .4f} {S_N1[2]: .4f}')
print(f'NODO 2: {S_N2[0]: .4f} {S_N2[1]: .4f} {S_N2[2]: .4f}')
print(f'NODO 3: {S_N3[0]: .4f} {S_N3[1]: .4f} {S_N3[2]: .4f}')

#We use the expresion for the Active power. 
P_N1 = abs(np.mean(P_ph1, axis=1))
P_N2 = abs(np.mean(P_ph2, axis=1))
P_N3 = abs(np.mean(P_ph3, axis=1))
#Total active power calculation
Tot_P = P_N1.sum() + P_N2.sum()+ P_N3.sum()
print("ACTIVE POWER [W]")
print(f'NODO 1: {P_N1[0]: .4f} {P_N1[1]: .4f} {P_N1[2]: .4f}')
print(f'NODO 2: {P_N2[0]: .4f} {P_N2[1]: .4f} {P_N2[2]: .4f}')
print(f'NODO 3: {P_N3[0]: .4f} {P_N3[1]: .4f} {P_N3[2]: .4f}')
print(f'P TOTAL: {Tot_P }')

# From S^2 = P^2 - Q^2 clear Q
Q_N1 = np.sqrt((S_N1**2)-(P_N1**2))
Q_N2 = np.sqrt((S_N2**2)-(P_N2**2))
Q_N3 = np.sqrt(abs((S_N3**2)-(P_N3**2)))
#Total active power calculation
Tot_Q = Q_N1.sum() + Q_N2.sum()+ Q_N3.sum()
print("REACTIVE POWER [VAr]")
print(f'NODO 1: {Q_N1[0]: .4f} {Q_N1[1]: .4f} {Q_N1[2]: .4f}')
print(f'NODO 2: {Q_N2[0]: .4f} {Q_N2[1]: .4f} {Q_N2[2]: .4f}')
print(f'NODO 3: {Q_N3[0]: .4f} {Q_N3[1]: .4f} {Q_N3[2]: .4f}')
print(f'P TOTAL: {Tot_Q}')

# Using the active power in relation with apparent power
PF_N1 = P_N1 / S_N1
PF_N2 = P_N2 / S_N2
PF_N3 = P_N3 / S_N3
print("POWER FACTOR")
print(f'NODO 1: {PF_N1[0]: .4f} {PF_N1[1]: .4f} {PF_N1[2]: .4f}')
print(f'NODO 2: {PF_N2[0]: .4f} {PF_N2[1]: .4f} {PF_N2[2]: .4f}')
print(f'NODO 3: {PF_N3[0]: .4f} {PF_N3[1]: .4f} {PF_N3[2]: .4f}')

# Power triangles for node 1
fig2, ax2 = plt.subplots(3, 1, figsize=(17, 12))
ax2[0].quiver(0, 0, P_N1[0], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax2[0].quiver(P_N1[0], 0, 0, Q_N1[0], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax2[0].quiver(0, 0, P_N1[0], Q_N1[0], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax2[0].set_ylim(-Q_N1[0]*0.1,Q_N1[0]*1.1 )
plt.axis([0, P_N1[0] + 10, -10, Q_N1[0] + 10])
ax2[0].set_xlim(-P_N1[0]*0.1,P_N1[0]*1.1 )
ax2[0].set_title("POWER TRIANGLE NODE 1, PHASE A")
ax2[1].quiver(0, 0, P_N1[1], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax2[1].quiver(P_N1[1], 0, 0, Q_N1[1], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax2[1].quiver(0, 0, P_N1[1], Q_N1[1], angles='xy', color=['red'], scale_units='xy', scale=1, label='APPARENT POWER')
ax2[1].set_ylim(-Q_N1[1]*0.1,Q_N1[1]*1.1 )
plt.axis([0, P_N1[1] + 10, -10, Q_N1[1] + 10])
ax2[1].set_xlim(-P_N1[1]*0.1,P_N1[1]*1.1 )
ax2[1].set_title("POWER TRIANGLE NODE 1, PHASE B")
ax2[2].quiver(0, 0, P_N1[2], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax2[2].quiver(P_N1[2], 0, 0, Q_N1[2], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax2[2].quiver(0, 0, P_N1[2], Q_N1[2], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax2[2].set_ylim(-Q_N1[2]*0.1,Q_N1[2]*1.1 )
plt.axis([0, P_N1[2] + 10, -10, Q_N1[2] + 10])
ax2[2].set_xlim(-P_N1[2]*0.1,P_N1[2]*1.1 )
ax2[2].set_title("POWER TRIANGLE NODE 1, PHASE C")
ax2[0].legend(loc='upper left')
ax2[0].grid()
ax2[1].legend(loc='upper left')
ax2[1].grid()
ax2[2].legend(loc='upper left')
ax2[2].grid()
plt.show()

# Power triangles for node 2
fig3, ax3 = plt.subplots(3, 1, figsize=(17, 12))
ax3[0].quiver(0, 0, P_N2[0], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax3[0].quiver(P_N2[0], 0, 0, Q_N2[0], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax3[0].quiver(0, 0, P_N2[0], Q_N2[0], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax3[0].set_ylim(-Q_N2[0]*0.1,Q_N2[0]*1.1 )
plt.axis([0, P_N2[0] + 10, -10, Q_N2[0] + 10])
ax3[0].set_xlim(-P_N2[0]*0.1,P_N2[0]*1.1 )
ax3[0].set_title("POWER TRIANGLE NODE 2, PHASE A")
ax3[1].quiver(0, 0, P_N2[1], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax3[1].quiver(P_N2[1], 0, 0, Q_N2[1], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax3[1].quiver(0, 0, P_N2[1], Q_N2[1], angles='xy', color=['red'], scale_units='xy', scale=1, label='APPARENT POWER')
ax3[1].set_ylim(-Q_N2[1]*0.1,Q_N2[1]*1.1 )
plt.axis([0, P_N2[1] + 10, -10, Q_N2[1] + 10])
ax3[1].set_xlim(-P_N2[1]*0.1,P_N2[1]*1.1 )
ax3[1].set_title("POWER TRIANGLE NODE 2, PHASE B")
ax3[2].quiver(0, 0, P_N2[2], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax3[2].quiver(P_N2[2], 0, 0, Q_N2[2], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax3[2].quiver(0, 0, P_N2[2], Q_N2[2], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax3[2].set_ylim(-Q_N2[2]*0.1,Q_N2[2]*1.1 )
plt.axis([0, P_N2[2] + 10, -10, Q_N2[2] + 10])
ax3[2].set_xlim(-P_N2[2]*0.1,P_N2[2]*1.1 )
ax3[2].set_title("POWER TRIANGLE NODE 2, PHASE C")
ax3[0].legend(loc='upper left')
ax3[0].grid()
ax3[1].legend(loc='upper left')
ax3[1].grid()
ax3[2].legend(loc='upper left')
ax3[2].grid()
plt.show()

# Power triangles for node 3
fig4, ax4 = plt.subplots(3, 1, figsize=(17, 12))
ax4[0].quiver(0, 0, P_N3[0], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax4[0].quiver(P_N3[0], 0, 0, Q_N3[0], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax4[0].quiver(0, 0, P_N3[0], Q_N3[0], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax4[0].set_ylim(-Q_N3[0]*0.1,Q_N3[0]*1.1 )
plt.axis([0, P_N3[0] + 10, -10, Q_N3[0] + 10])
ax4[0].set_xlim(-P_N3[0]*0.1,P_N3[0]*1.1 )
ax4[0].set_title("POWER TRIANGLE NODE 3, PHASE A")
ax4[1].quiver(0, 0, P_N3[1], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax4[1].quiver(P_N3[1], 0, 0, Q_N3[1], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax4[1].quiver(0, 0, P_N3[1], Q_N3[1], angles='xy', color=['red'], scale_units='xy', scale=1, label='APPARENT POWER')
ax4[1].set_ylim(-Q_N3[1]*0.1,Q_N3[1]*1.1 )
plt.axis([0, P_N3[1] + 10, -10, Q_N3[1] + 10])
ax4[1].set_xlim(-P_N3[1]*0.1,P_N3[1]*1.1 )
ax4[1].set_title("POWER TRIANGLE NODE 3, PHASE B")
ax4[2].quiver(0, 0, P_N3[2], 0, angles='xy', color=['green'], scale_units='xy', scale=1, label='ACTIVE POWER')
ax4[2].quiver(P_N3[2], 0, 0, Q_N3[2], angles='xy', color=['purple'], scale_units='xy', scale=1, label='REACTIVE POWER')
ax4[2].quiver(0, 0, P_N3[2], Q_N3[2], angles='xy', color=['red'], scale_units='xy', scale=1, label='APARENT POWER')
ax4[2].set_ylim(-Q_N3[2]*0.1,Q_N3[2]*1.1 )
plt.axis([0, P_N3[2] + 10, -10, Q_N3[2] + 10])
ax4[2].set_xlim(-P_N3[2]*0.1,P_N3[2]*1.1 )
ax4[2].set_title("POWER TRIANGLE NODE 3, PHASE C")
ax4[0].legend(loc='upper left')
ax4[0].grid()
ax4[1].legend(loc='upper left')
ax4[1].grid()
ax4[2].legend(loc='upper left')
ax4[2].grid()
plt.show()

# Voltage phase diagram
F = 60
Vn1theta_a = (np.argmax(voltage_1[0])-np.argmax(voltage_1[0]))*(1/(100*F))*2*np.pi*F
Vn1theta_b = (np.argmax(voltage_1[0])-np.argmax(voltage_1[1]))*(1/(100*F))*2*np.pi*F
Vn1theta_c = (np.argmax(voltage_1[0])-np.argmax(voltage_1[2]))*(1/(100*F))*2*np.pi*F
Vn2theta_a = (np.argmax(voltage_1[0])-np.argmax(voltage_2[0]))*(1/(100*F))*2*np.pi*F
Vn2theta_b = (np.argmax(voltage_1[0])-np.argmax(voltage_2[1]))*(1/(100*F))*2*np.pi*F
Vn2theta_c = (np.argmax(voltage_1[0])-np.argmax(voltage_2[2]))*(1/(100*F))*2*np.pi*F
Vn3theta_a = (np.argmax(voltage_1[0])-np.argmax(voltage_3[0]))*(1/(100*F))*2*np.pi*F
Vn3theta_b = (np.argmax(voltage_1[0])-np.argmax(voltage_3[1]))*(1/(100*F))*2*np.pi*F
Vn3theta_c = (np.argmax(voltage_1[0])-np.argmax(voltage_3[2]))*(1/(100*F))*2*np.pi*F

# Graphing voltage phases diagram
fig5, ax5 = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(16, 16))
ax5[0].quiver(Vn1theta_a, np.max(voltage_1[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax5[0].quiver(Vn1theta_b, np.max(voltage_1[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax5[0].quiver(Vn1theta_c, np.max(voltage_1[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax5[0].set_rmax(np.max(voltage_1))
ax5[0].legend(loc="lower right")
ax5[0].set_title('VOLTAGES PHASE DIAGRAM NODE 1')
ax5[1].quiver(Vn2theta_a, np.max(voltage_2[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax5[1].quiver(Vn2theta_b, np.max(voltage_2[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax5[1].quiver(Vn2theta_c, np.max(voltage_2[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax5[1].set_rmax(np.max(voltage_2))
ax5[1].legend(loc="lower right")
ax5[1].set_title('VOLTAGES PHASE DIAGRAM NODE 2')
ax5[2].quiver(Vn3theta_a, np.max(voltage_3[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax5[2].quiver(Vn3theta_b, np.max(voltage_3[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax5[2].quiver(Vn3theta_c, np.max(voltage_3[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax5[2].set_rmax(np.max(voltage_3))
ax5[2].legend(loc="lower right")
ax5[2].set_title('VOLTAGES PHASE DIAGRAM NODE 3')

# Stablishing phasors of voltage
V1fasor_a = np.max(voltage_1[0]) * np.exp(1j * Vn1theta_a)
V1fasor_b = np.max(voltage_1[1]) * np.exp(1j * Vn1theta_b)
V1fasor_c = np.max(voltage_1[2]) * np.exp(1j * Vn1theta_c)
V2fasor_a = np.max(voltage_2[0]) * np.exp(1j * Vn2theta_a)
V2fasor_b = np.max(voltage_2[1]) * np.exp(1j * Vn2theta_b)
V2fasor_c = np.max(voltage_2[2]) * np.exp(1j * Vn2theta_c)
V3fasor_a = np.max(voltage_3[0]) * np.exp(1j * Vn3theta_a)
V3fasor_b = np.max(voltage_3[1]) * np.exp(1j * Vn3theta_b)
V3fasor_c = np.max(voltage_3[2]) * np.exp(1j * Vn3theta_c)
print(" ")
print(f"Voltage fasor node 1 phase A, magnitude: {np.abs(V1fasor_a): .4f}, angle(degrees): {np.angle(V1fasor_a)*180/np.pi: .4f}")
print(f"Voltage fasor node 1 phase B, magnitude: {np.abs(V1fasor_b): .4f}, angle(degrees): {np.angle(V1fasor_b)*180/np.pi: .4f}")
print(f"Voltage fasor node 1 phase C, magnitude: {np.abs(V1fasor_c): .4f}, angle(degrees): {np.angle(V1fasor_c)*180/np.pi: .4f}")
print(" ")
print(f"Voltage fasor node 2 phase A, magnitude: {np.abs(V2fasor_a): .4f}, angle(degrees): {np.angle(V2fasor_a)*180/np.pi: .4f}")
print(f"Voltage fasor node 2 phase B, magnitude: {np.abs(V2fasor_b): .4f}, angle(degrees): {np.angle(V2fasor_b)*180/np.pi: .4f}")
print(f"Voltage fasor node 2 phase C, magnitude: {np.abs(V2fasor_c): .4f}, angle(degrees): {np.angle(V2fasor_c)*180/np.pi: .4f}")
print(" ")
print(f"Voltage fasor node 3 phase A, magnitude: {np.abs(V3fasor_a): .4f}, angle(degrees): {np.angle(V3fasor_a)*180/np.pi: .4f}")
print(f"Voltage fasor node 3 phase B, magnitude: {np.abs(V3fasor_b): .4f}, angle(degrees): {np.angle(V3fasor_b)*180/np.pi: .4f}")
print(f"Voltage fasor node 3 phase C, magnitude: {np.abs(V3fasor_c): .4f}, angle(degrees): {np.angle(V3fasor_c)*180/np.pi: .4f}")

# Current phase diagram
F = 60
In1theta_a = (np.argmax(voltage_1[0])-np.argmax(current_1[0]))*(1/(100*F))*2*np.pi*F
In1theta_b = (np.argmax(voltage_1[0])-np.argmax(current_1[1]))*(1/(100*F))*2*np.pi*F
In1theta_c = (np.argmax(voltage_1[0])-np.argmax(current_1[2]))*(1/(100*F))*2*np.pi*F
In2theta_a = (np.argmax(voltage_1[0])-np.argmax(current_2[0]))*(1/(100*F))*2*np.pi*F
In2theta_b = (np.argmax(voltage_1[0])-np.argmax(current_2[1]))*(1/(100*F))*2*np.pi*F
In2theta_c = (np.argmax(voltage_1[0])-np.argmax(current_2[2]))*(1/(100*F))*2*np.pi*F
In3theta_a = (np.argmax(voltage_1[0])-np.argmax(current_3[0]))*(1/(100*F))*2*np.pi*F
In3theta_b = (np.argmax(voltage_1[0])-np.argmax(current_3[1]))*(1/(100*F))*2*np.pi*F
In3theta_c = (np.argmax(voltage_1[0])-np.argmax(current_3[2]))*(1/(100*F))*2*np.pi*False

# Graphing current phase diagram
fig6, ax6 = plt.subplots(1, 3, subplot_kw={'projection': 'polar'}, figsize=(16, 16))
ax6[0].quiver(In1theta_a, np.max(current_1[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax6[0].quiver(In1theta_b, np.max(current_1[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax6[0].quiver(In1theta_c, np.max(current_1[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax6[0].set_rmax(np.max(current_1))
ax6[0].legend(loc="lower right")
ax6[0].set_title('CURRENT PHASE DIAGRAM NODE 1')
ax6[1].quiver(In2theta_a, np.max(current_2[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax6[1].quiver(In2theta_b, np.max(current_2[1]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax6[1].quiver(In2theta_c, np.max(current_2[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax6[1].set_rmax(np.max(current_2))
ax6[1].legend(loc="lower right")
ax6[1].set_title('CURRENT PHASE DIAGRAM NODE 2')
ax6[2].quiver(In3theta_a, np.max(current_3[0]), angles='xy',  scale_units='xy', scale=1, color='green', label='PHASE A')
ax6[2].quiver(In3theta_b, np.max(current_3[2]), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax6[2].quiver(In3theta_c, np.max(current_3[2]), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax6[2].set_rmax(np.max(current_3))
ax6[2].legend(loc="lower right")
ax6[2].set_title('CURRENT PHASE DIAGRAM NODE 3')

# Stablishing phasors of current
I1fasor_a = np.max(current_1[0]) * np.exp(1j * In1theta_a)
I1fasor_b = np.max(current_1[1]) * np.exp(1j * In1theta_b)
I1fasor_c = np.max(current_1[2]) * np.exp(1j * In1theta_c)
I2fasor_a = np.max(current_2[0]) * np.exp(1j * In2theta_a)
I2fasor_b = np.max(current_2[1]) * np.exp(1j * In2theta_b)
I2fasor_c = np.max(current_2[2]) * np.exp(1j * In2theta_c)
I3fasor_a = np.max(current_3[0]) * np.exp(1j * In3theta_a)
I3fasor_b = np.max(current_3[1]) * np.exp(1j * In3theta_b)
I3fasor_c = np.max(current_3[2]) * np.exp(1j * In3theta_c)
print(f"Current fasor node 1 phase A, magnitude: {np.abs(I1fasor_a): .4f}, angle(degrees): {np.angle(I1fasor_a)*180/np.pi: .4f}")
print(f"Current fasor node 1 phase B, magnitude: {np.abs(I1fasor_b): .4f}, angle(degrees): {np.angle(I1fasor_b)*180/np.pi: .4f}")
print(f"Current fasor node 1 phase C, magnitude: {np.abs(I1fasor_c): .4f}, angle(degrees): {np.angle(I1fasor_c)*180/np.pi: .4f}")
print(" ")
print(f"Current fasor node 2 phase A, magnitude: {np.abs(I2fasor_a): .4f}, angle(degrees): {np.angle(I2fasor_a)*180/np.pi: .4f}")
print(f"Current fasor node 2 phase B, magnitude: {np.abs(I2fasor_b): .4f}, angle(degrees): {np.angle(I2fasor_b)*180/np.pi: .4f}")
print(f"Current fasor node 2 phase C, magnitude: {np.abs(I2fasor_c): .4f}, angle(degrees): {np.angle(I2fasor_c)*180/np.pi: .4f}")
print(" ")
print(f"Current fasor node 3 phase A, magnitude: {np.abs(I3fasor_a): .4f}, angle(degrees): {np.angle(I3fasor_a)*180/np.pi: .4f}")
print(f"Current fasor node 3 phase B, magnitude: {np.abs(I3fasor_b): .4f}, angle(degrees): {np.angle(I3fasor_b)*180/np.pi: .4f}")
print(f"Current fasor node 3 phase C, magnitude: {np.abs(I3fasor_c): .4f}, angle(degrees): {np.angle(I3fasor_c)*180/np.pi: .4f}")

# Calculating current and voltage at node 4

# Wire impedances
Zl1 = 0.009j
Zl2 = 0.01j
Zl3 = 0.01 + 0.001j
# Current per phase at node 4
I4fasor_a = I1fasor_a + I2fasor_a + I3fasor_a
I4fasor_b = I1fasor_b + I2fasor_b + I3fasor_b
I4fasor_c = I1fasor_c + I2fasor_c + I3fasor_c
# Voltage per phase at node 4
V4fasor_a = ((V1fasor_a/Zl1) + (V2fasor_a/Zl2) + (V3fasor_a/Zl3) + (I4fasor_a) ) / ( (1/Zl1) + (1/Zl2) + (1/Zl3))
V4fasor_b = ((V1fasor_b/Zl1) + (V2fasor_b/Zl2) + (V3fasor_b/Zl3) + (I4fasor_b) ) / ( (1/Zl1) + (1/Zl2) + (1/Zl3))
V4fasor_c = ((V1fasor_c/Zl1) + (V2fasor_c/Zl2) + (V3fasor_c/Zl3) + (I4fasor_c) ) / ( (1/Zl1) + (1/Zl2) + (1/Zl3))
print(" ")
print(f"Voltage fasor node 4 phase A, magnitude: {np.abs(V4fasor_a): .4f}, angle(degrees): {np.angle(V4fasor_a)*180/np.pi: .4f}")
print(f"Voltage fasor node 4 phase B, magnitude: {np.abs(V4fasor_b): .4f}, angle(degrees): {np.angle(V4fasor_b)*180/np.pi: .4f}")
print(f"Voltage fasor node 4 phase C, magnitude: {np.abs(V4fasor_c): .4f}, angle(degrees): {np.angle(V4fasor_c)*180/np.pi: .4f}")
print(" ")
print(f"Current fasor node 4 phase A, magnitude: {np.abs(I4fasor_a): .4f}, angle(degrees): {np.angle(I4fasor_a)*180/np.pi: .4f}")
print(f"Current fasor node 4 phase B, magnitude: {np.abs(I4fasor_b): .4f}, angle(degrees): {np.angle(I4fasor_b)*180/np.pi: .4f}")
print(f"Current fasor node 4 phase C, magnitude: {np.abs(I4fasor_c): .4f}, angle(degrees): {np.angle(I4fasor_c)*180/np.pi: .4f}")

# Graphing voltage and current phase diagram for node 4
fig7, ax7 = plt.subplots(1, 2, subplot_kw={'projection': 'polar'}, figsize=(12, 8))
ax7[0].quiver(np.angle(V4fasor_a), np.abs(V4fasor_a), angles='xy', scale_units='xy', scale=1, color='green', label='PHASE A')
ax7[0].quiver(np.angle(V4fasor_b), np.abs(V4fasor_b), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax7[0].quiver(np.angle(V4fasor_c), np.abs(V4fasor_c), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax7[0].set_rmax(np.abs(V4fasor_a))
ax7[0].legend(loc="lower right")
ax7[0].set_title('VOLTAGES PHASE DIAGRAM NODE 4')
ax7[1].quiver(np.angle(I4fasor_a), np.abs(I4fasor_a), angles='xy', scale_units='xy', scale=1, color='green', label='PHASE A')
ax7[1].quiver(np.angle(I4fasor_b), np.abs(I4fasor_b), angles='xy', scale_units='xy', scale=1, color='blue', label='PHASE B')
ax7[1].quiver(np.angle(I4fasor_c), np.abs(I4fasor_c), angles='xy', scale_units='xy', scale=1, color='red', label='PHASE C')
ax7[1].set_rmax(np.abs(I4fasor_a))
ax7[1].legend(loc="lower right")
ax7[1].set_title('CURRENT PHASE DIAGRAM NODE 4')

# Complex power
S4com_a = V4fasor_a * I4fasor_a.conjugate()
S4com_b = V4fasor_b * I4fasor_b.conjugate()
S4com_c = V4fasor_c * I4fasor_c.conjugate()

# Active power, Reactive power, Apparent power and Power factor at node 4
P4_a = np.real(S4com_a)
P4_b = np.real(S4com_b)
P4_c = np.real(S4com_c)
Q4_a = np.imag(S4com_a)
Q4_b = np.imag(S4com_b)
Q4_c = np.imag(S4com_c)
S4_a = np.sqrt(P4_a**2 + Q4_a**2)
S4_b = np.sqrt(P4_b**2 + Q4_b**2)
S4_c = np.sqrt(P4_c**2 + Q4_c**2)
FP4_a = P4_a / S4_a
FP4_b = P4_b / S4_b
FP4_c = P4_c / S4_c
print(" ")
print("Active power at node 4 [W]")
print(f"P phase A: {P4_a: .4f}")
print(f"P phase B: {P4_b: .4f}")
print(f"P phase C: {P4_c: .4f}")
print("Reactive power at node 4 [VAr]")
print(f"Q phase A: {Q4_a: .4f}")
print(f"Q phase B: {Q4_b: .4f}")
print(f"Q phase C: {Q4_c: .4f}")
print("Apparent power at node 4 [VA]")
print(f"S phase A: {S4_a: .4f}")
print(f"S phase B: {S4_b: .4f}")
print(f"S phase C: {S4_c: .4f}")
print("Power factor")
print(f"FP phase A: {FP4_a: .4f}")
print(f"FP phase B: {FP4_b: .4f}")
print(f"FP phase C: {FP4_c: .4f}")

# Time vector
F = 60
# Sample time
Ts = 1/(100*F)
t = np.arange(0.0, 5 / 60, Ts)
# Sinusoidal voltage function
V4_at = np.abs(V4fasor_a) * np.cos(2 * np.pi * 60 * t + np.angle(V4fasor_a))
V4_bt = np.abs(V4fasor_b) * np.cos(2 * np.pi * 60 * t + np.angle(V4fasor_b))
V4_ct = np.abs(V4fasor_c) * np.cos(2 * np.pi * 60 * t + np.angle(V4fasor_c))
# Sinusoidal current function
I4_at = np.abs(I4fasor_a) * np.cos(2 * np.pi * 60 * t + np.angle(I4fasor_a))
I4_bt = np.abs(I4fasor_b) * np.cos(2 * np.pi * 60 * t + np.angle(I4fasor_b))
I4_ct = np.abs(I4fasor_c) * np.cos(2 * np.pi * 60 * t + np.angle(I4fasor_c))
# Graphing voltage and current in time domain at node 4
fig8, ax8 = plt.subplots(2, 1, figsize=(14, 8))
# Voltage per phase at node 4
ax8[0].plot(t, V4_at)
ax8[0].plot(t, V4_bt)
ax8[0].plot(t, V4_ct)
ax8[0].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax8[0].set_title("Voltage at node 4")
ax8[0].grid()
# Current per phase at node 4
ax8[1].plot(t, I4_at)
ax8[1].plot(t, I4_bt)
ax8[1].plot(t, I4_ct)
ax8[1].legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax8[1].set_title("Current at node 4")
ax8[1].grid()
plt.show()

# Calculation of instantaneous power
P_ph4_a = V4_at * I4_at
P_ph4_b = V4_bt * I4_bt
P_ph4_c = V4_ct * I4_ct
# Graphing voltage and current in time domain at node 4
fig8, ax9 = plt.subplots(1, 1, figsize=(14, 4))
# Voltage per phase at node 4
ax9.plot(P_ph4_a.T)
ax9.plot(P_ph4_b.T)
ax9.plot(P_ph4_c.T)
ax9.legend(["Phase A", "Phase B", "Phase C"], loc='upper right')
ax9.set_title("Instantaneous power at node 4")
ax9.grid()
plt.show()