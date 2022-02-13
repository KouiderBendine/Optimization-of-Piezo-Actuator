from scipy.signal import cont2discrete, lti, dlti, dstep
import matplotlib.pyplot as plt
import numpy as np
import csv
import os
import xlsxwriter
# DEFINE CONSTANTS
#ctr = 'Control_off'    #  if the controller is off
ctr = 'Control_on'    #  if the controller is on
qessi = 0.019    # Damping ratio
N_Mode = 6       # Mode number
## Read Txt file from Numerical test

data = np.loadtxt('3Tran_Analysis.txt', delimiter=",")

## Import Mechanical Load
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
    Total_vec = np.zeros((4, 12))
    for line in csv_reader:
        Total_vec[i, :] = np.matrix(line)
        i = i + 1

B_vect = Total_vec[:, 0:N_Mode]
A_vect = Total_vec[:, N_Mode:12]

## Calculate The fitness function

LamdaS = 0
LamdaP = 1
for N in range(N_Mode):
    LamdaS = LamdaS + np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[1, N])
    LamdaP = LamdaP * np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[1, N])
Fitn = LamdaS * (LamdaP) ** (1 / N_Mode)
print(Fitn)


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
AF = 1
AC = 100
N  = 5
HG  = 2*N/f
Tstop =3 # 2*N/f
TIM = N/f/100

# define the continuous-time system matrices

A, B1, B_F = getB(B_vect, A_vect, Force_vec)

# define the number of time-samples used for the simulation and the sampling time for the discretization

time = 600
sampling = 1
dt = 0.005


if ctr == 'Control_off':
    d_system = cont2discrete((A, B_F, B_F.T, [0]), dt)
else:
    d_system = cont2discrete((A, B1, B1.T, [0]), dt)

Ad = d_system[0]    # A matrix for discret state space sytem
Bd = d_system[1]    # B matrix for discret state space sytem
Xd = np.zeros(shape=(A.shape[0], time + 1))
actual_state_x = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
x = actual_state_x

for i in range(time):
    Timestep = (i + 1) * TIM
    if ctr == 'Control_off':
        x = np.dot(Ad, x) + Bd*AF*np.sin(2*np.pi*6*Timestep)
        Xd[:, [i]] = x[:, [0]]
    else:
        x = np.dot(Ad, x) + np.array([Bd[:, 2]]).T * AC * np.sin(2 * np.pi * 6 * Timestep)
        Xd[:, [i]] = np.array([x[0, :]]).T

plt.plot(Xd[0, 0:200], label="State Space", marker='o', color='k')
plt.plot(data[0:200, 0], label="Ansys Simulation", linewidth=4, color='k')
plt.xlabel('Time simpling', fontsize=18)
plt.ylabel('Dispalcement (m)', fontsize=18)
#plt.title('5 tune burst')
plt.legend(loc="upper left")
plt.grid()

B = data
#### Save the Fitness Functions
workbook = xlsxwriter.Workbook('ValidationAct3.xlsx')

worksheet = workbook.add_worksheet()
col = 0
for row, data in enumerate(np.transpose(Xd)):
    worksheet.write_column(row, col, data)
col = 1
for row, data in enumerate(B):
     worksheet.write_column(row, col, data)




workbook.close()
