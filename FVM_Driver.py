# Third party libraries
import numpy as np
from matplotlib import pyplot as plt
import os

# custom flux-splitting schemes
from AUSM import AUSM
from VanLeer import VanLeer
from SW import SW

#-------------------------------------------#
#        Shock Tube Problem Settings        #
#-------------------------------------------#
# Gas Constants
gamma = 1.4
R = 287

# Test options -> choose Flux scheme, test to run and output
splitting = "AUSM"  # Steger_Warming/Van_Leer/AUSM
Test = "Test_1"  # Test_1/Test_2/Test_3/Test_4
visualize = "density"  # density/velocity/pressure/energy
Nc = 100  # Number of cells (Resolution)


#-------------------------------------------#
#               Miscellaneous               #
#-------------------------------------------#
if Test == "Test_1":
    # Shock Tube Initial States
    # LEFT
    l_rho = 1
    l_u = 0.75
    l_p = 1
    l_e = l_p / ((gamma - 1) * l_rho)  # Left-side specific internal energy

    # RIGHT
    r_rho = 0.125
    r_u = 0
    r_p = 0.1
    r_e = r_p / ((gamma - 1) * r_rho)  # Right-side specific internal energy

    x_0 = 0.3  # Position of initial shock
    t_end = 0.2  # t of evaluation

elif Test == "Test_2":
    # Shock Tube Initial States
    # LEFT
    l_rho = 1
    l_u = -2
    l_p = 0.4
    l_e = l_p / ((gamma - 1) * l_rho)  # Left-side specific internal energy

    # RIGHT
    r_rho = 1
    r_u = 2
    r_p = 0.4
    r_e = r_p / ((gamma - 1) * r_rho)  # Right-side specific internal energy

    x_0 = 0.5  # Position of initial shock
    t_end = 0.15  # t of evaluation

elif Test == "Test_3":
    # Shock Tube Initial States
    # LEFT
    l_rho = 1
    l_u = 0
    l_p = 1000
    l_e = l_p / ((gamma - 1) * l_rho)  # Left-side specific internal energy

    # RIGHT
    r_rho = 1
    r_u = 0
    r_p = 0.01
    r_e = r_p / ((gamma - 1) * r_rho)  # Right-side specific internal energy

    x_0 = 0.5  # Position of initial shock
    t_end = 0.012  # t of evaluation

elif Test == "Test_4":
    # Shock Tube Initial States
    # LEFT
    l_rho = 5.99924
    l_u = 19.5975
    l_p = 460.894
    l_e = l_p / ((gamma - 1) * l_rho)  # Left-side specific internal energy

    # RIGHT
    r_rho = 5.99242
    r_u = -6.19633
    r_p = 46.0950
    r_e = r_p / ((gamma - 1) * r_rho)  # Right-side specific internal energy

    x_0 = 0.4  # Position of initial shock
    t_end = 0.035  # t of evaluation
else:
    print("Test case not set properly. Check spelling")

#-------------------------------------------#
#                   Grid                    #
#-------------------------------------------#
# Spatial
x_min = 0
x_max = 1

# Grid step
dx = (x_max - x_min) / Nc  # Grid step based on given data

# Grid Vector
x = np.linspace(x_min, x_max, Nc)

#-------------------------------------------#
#            Initial State Vector           #
#-------------------------------------------#
# Conserved variables, state vector
U1 = np.zeros(Nc)
U2 = np.zeros(Nc)
U3 = np.zeros(Nc)

for i in range(Nc):
    if x[i] < x_0:
        U1[i] = l_rho
        U2[i] = l_rho * l_u
        U3[i] = l_rho * (l_e + 0.5 * l_u ** 2)
    else:
        U1[i] = r_rho
        U2[i] = r_rho * r_u
        U3[i] = r_rho * (r_e + 0.5 * r_u ** 2)

#-------------------------------------------#
#                   Solver                  #
#-------------------------------------------#
t = 0        # time variable allocation
loop = 0     # Counter variable initialization/allocation

if splitting == "Van_Leer":
    CFL = 0.6
else:
    CFL = 0.9

while t < t_end:
    # Initializing t-1 timestep
    U1_old = U1  # U1 [rho]
    U2_old = U2  # U2 [rho * u]
    U3_old = U3  # U3 [E]

    # Obtaining variables in terms of state-vector conserved quantities
    p = (gamma - 1) * (U3_old - 0.5 * ((U2_old ** 2) / U1_old))  # Pressure
    a = np.sqrt((gamma * p) / U1_old)  # Speed of sound
    u = U2_old / U1_old  # x-velocity
    H = (U3_old + p) / U1_old  # Total specific enthalpy
    M = u / a

    # CFL adjustment for first 5 steps
    if loop < 5:
        # Getting time-step based on CFL
        dt = (dx * CFL / (max(abs(u) + a))) * 0.2
    else:
        dt = dx * CFL / (max(abs(u) + a))

    # Choice of Flux Splitting -> Outputs are Inter-Cell Fluxes
    if splitting == "Van_Leer":
        F1_p, F2_p, F3_p, F1_n, F2_n, F3_n = VanLeer(U1_old, a, M, u, gamma)
    elif splitting == "AUSM":
        F1_p, F2_p, F3_p, F1_n, F2_n, F3_n = AUSM(U1_old, a, p, M, u, H)
    elif splitting == "Steger_Warming":
        F1_p, F2_p, F3_p, F1_n, F2_n, F3_n = SW(U1_old, gamma, a, u, H)
    else:
        print("Flux Splitting Initialization Error. Check spelling")

    # Inter-face Fluxes
    F1 = F1_p[0:(len(F1_p) - 1)] + F1_n[1:len(F1_n)]
    F2 = F2_p[0:(len(F2_p) - 1)] + F2_n[1:len(F2_n)]
    F3 = F3_p[0:(len(F3_p) - 1)] + F3_n[1:len(F3_n)]

    # Time marching
    U1[1:(len(U1) - 1)] = U1_old[1:(len(U1_old)) - 1] - (dt / dx) * (F1[1:len(F1)] - F1[0:(len(F1) - 1)])
    U2[1:(len(U1) - 1)] = U2_old[1:(len(U2_old)) - 1] - (dt / dx) * (F2[1:len(F2)] - F2[0:(len(F2) - 1)])
    U3[1:(len(U1) - 1)] = U3_old[1:(len(U3_old)) - 1] - (dt / dx) * (F3[1:len(F3)] - F3[0:(len(F3) - 1)])

    # Transmissive Boundary Condition
    U1[0] = U1[1]
    U2[0] = U2[1]
    U3[0] = U3[1]
    U1[-1] = U1[-2]
    U2[-1] = U2[-2]
    U3[-1] = U3[-2]

    # Updated Pressure
    p_new = (gamma - 1) * (U3 - 0.5 * ((U2 ** 2) / U1))
    # Updated Internal Energy
    Ene_new = p_new/((gamma-1)*U1)
    # Updated Velocity
    U_new = U2 / U1
    # Updated Density
    Rho_new = U1

    # Counters
    loop = loop + 1
    t = t + dt

# Plotting
CWD = os.getcwd()  # Path to working directory
path = CWD + "/ToroResults/" + Test + "/" + visualize + ".csv"
my_data = np.genfromtxt(path, delimiter=',')
plt.plot(my_data[:, 0], my_data[:, 1], label='Analytic', color='black')

if visualize == "density":
    plt.scatter(x, U1, label=splitting, color='black')
elif visualize == "velocity":
    plt.scatter(x, U2 / U1, label=splitting, color='black')
elif visualize == "pressure":
    plt.scatter(x, p_new, label=splitting, color='black')
elif visualize == "energy":
    plt.scatter(x, Ene_new, label=splitting, color='black')
else:
    print("Variable to plot not specified properly. Check spelling")

plt.axvline(x_0, color='k', linestyle='--', label='x_0')
plt.xlabel('Distance from origin (m)')
plt.ylabel(visualize)
plt.xlim([0, 1])
plt.title(Test)
plt.legend()
plt.show()
