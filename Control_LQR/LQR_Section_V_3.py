from scipy.signal import cont2discrete, lti, dlti, dstep
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import control
import xlsxwriter
from random import gauss
# DEFINE CONSTANTS
qessi = 0.019    # Damping ratio
N_Mode = 6       # Mode number
## Read Txt file from Numerical test


## Import Mechanical Load

with open(os.path.join("D:\Vibration_control_Paper\Ansys_Code\GA_Algo_Patc", "State_Space_Force.csv"), 'r') as file:
    csv_reader = csv.reader(file, delimiter=",", quotechar='|')

    for line in csv_reader:
        Load_vec = np.matrix(line)
Force_vec = np.zeros((1, N_Mode))

for k in range(N_Mode):
    Force_vec[0, k] = float(Load_vec[0, k])

## Import Frequencies and electric Load

with open(os.path.join("D:\Vibration_control_Paper\Ansys_Code\GA_Algo_Patc", "State_Space.csv"), 'r') as file:
    csv_reader = csv.reader(file, delimiter=",", quotechar='|')
    i = 0
    Total_vec = np.zeros((5, 12))
    for line in csv_reader:
        Total_vec[i, :] = np.matrix(line)
        i = i + 1

B_vect = Total_vec[0:4, 0:N_Mode]
A_vect = Total_vec[0:4, N_Mode:12]

def getB(B_vect, A_vect, Force_vec):
    """
    Return:
    MATRIX A   The state matrix
    MATRIX B   The Actuators control inputs
    """
    B = np.concatenate((np.zeros((N_Mode, 4)), np.transpose(B_vect)), axis=0)
    B_F = np.concatenate((np.zeros((N_Mode, 1)), np.transpose(Force_vec)), axis=0)

    A1 = np.array([[2 * np.pi*A_vect[0, 0], 0, 0, 0, 0, 0],
                   [0,  2 * np.pi*A_vect[0, 1], 0, 0, 0, 0],
                   [0, 0,  2 * np.pi*A_vect[0, 2], 0, 0, 0],
                   [0, 0, 0,  2 * np.pi*A_vect[0, 3], 0, 0],
                   [0, 0, 0, 0,  2 * np.pi*A_vect[0, 4], 0],
                   [0, 0, 0, 0, 0,  2 * np.pi*A_vect[0, 5]]])

    A2 = np.array([[-qessi * 4 * np.pi* A_vect[0, 0], 0, 0, 0, 0, 0],
                   [0, -qessi * 4 * np.pi * A_vect[0, 1], 0, 0, 0, 0],
                   [0, 0, -qessi * 4 * np.pi * A_vect[0, 2], 0, 0, 0],
                   [0, 0, 0, -qessi * 4 * np.pi * A_vect[0, 3], 0, 0],
                   [0, 0, 0, 0, -qessi * 4 * np.pi * A_vect[0, 4], 0],
                   [0, 0, 0, 0, 0, -qessi * 4 * np.pi * A_vect[0, 5]]])


    A11 = np.concatenate((np.zeros((N_Mode, N_Mode)), A1), axis=1)
    A22 = np.concatenate((-A1, A2), axis=1)
    A = np.concatenate((A11, A22), axis=0)

    return A, B, B_F

f = 10
AF = 0.1
AC = 100
N  = 5
HG  = 2*N/f
Tstop = 3 # 2*N/f
TIM = 0.005

# define the continuous-time system matrices

A, B1, B_F = getB(B_vect, A_vect, Force_vec)

# define the number of time-samples used for the simulation and the sampling time for the discretization

time = 600
sampling = 1
dt = 0.005

d_system_without = cont2discrete((A, B_F, B_F.T, [0]), dt)

d_system_with = cont2discrete((A, B1, B1.T, [0]), dt)
Q = 1e9*np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
R = 1e-4*np.diag([1, 1, 1, 1])


Ad = d_system_with[0]         # A matrix for discret state space sytem
Bd = d_system_with[1]         # B matrix for discret state space sytem
Bd_F = d_system_without[1]    # B matrix for discret state space sytem

RanF = [gauss(0.5, 1) for i in range(600)]
KLqr, X_S, E = control.dlqr(Ad, Bd, Q, R)

Xd = np.zeros(shape=(A.shape[0], time + 1))
Xdwith = np.zeros(shape=(A.shape[0], time + 1))
actual_state_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x = actual_state_x
x0 = actual_state_x
Uact = np.zeros(shape=(Bd.shape[1], time + 1))
for i in range(time):
    Timestep = (i + 1) * TIM

    #x0 = np.dot(Ad, x) + Bd_F*AF*np.sin(2*np.pi*6*Timestep)

    if i > 100 and i< 300:
        x0 = (Ad - Bd  @ KLqr) @ x0 + Bd_F*AF*np.sin(2*np.pi*6*Timestep)
        #x0 = (Ad - Bd @ KLqr) @ x0 + Bd_F * AF* RanF[i]
        Xdwith[:, i] = x0[0, :].T
        Uact[:, i] = -KLqr @ x0[0, :]
    else:
        x0 = Ad  @ x0 + Bd_F * AF * np.sin(2 * np.pi * 6 * Timestep)
        #x0 = Ad @ x0 + Bd_F * AF*RanF[i]
        Xdwith[:, i] = x0[0, :].T


    x = np.dot(Ad, x) + Bd_F*AF*np.sin(2 * np.pi * 6 * Timestep)
    #x = Ad @ x + Bd_F * AF * RanF[i]
    Xd[:, [i]] = np.array([x[0, :]]).T

plt.plot(Xd[0, 0:600], label="Without Control", marker='o', color='r')
plt.plot(Xdwith[0, 0:600], label="With Control", marker='o', color='b')

plt.xlabel('Time simpling', fontsize=18)
plt.ylabel('Dispalcement (m)', fontsize=18)

plt.legend(loc="upper left")
plt.grid()

plt.figure(2)
plt.plot(Uact[0, 0:600], label="Atuator 1", marker='o', color='k')
plt.plot(Uact[1, 0:600], label="Atuator 2", marker='*', color='r')
plt.plot(Uact[2, 0:600], label="Atuator 3", marker='o', color='b')
plt.plot(Uact[3, 0:600], label="Atuator 4", marker='*', color='g')
plt.xlabel('Time simpling', fontsize=18)
plt.ylabel('Voltage (V)', fontsize=18)
plt.legend(loc="upper left")
plt.grid()

#### Save Data
Dsp = np.array([Xd[0, 0:600], Xdwith[0, 0:600]])
workbook = xlsxwriter.Workbook('GaussForceDisp_Plate.xlsx')

worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(Dsp):
    worksheet.write_column(row, col, data)



workbook.close()

workbook = xlsxwriter.Workbook('GaussForceActVoltage_Plate.xlsx')

worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(Uact):
    worksheet.write_column(row, col, data)



workbook.close()