import subprocess
import csv
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter


qessi = 0.019
N_Mode = 6
Pos_vect = np.zeros((18, 8))
Fitness = np.zeros((18,  8))
B_total = np.zeros((4,  7, 8*18))
count = 0
## Start The FEM iteration analysis
for k in range(1, 8):
    for it in range(18):
        print('Runtine', k)
        Pos_vect[it, k] = it

        file1 = open("Iter_param.inp", "w+")
        file1.write('ZC=%e\n' % k)
        file1.write('YC=%e\n' % it)
        file1.write('iter=%e\n' % k)
        file1.close()
        cmd = ["C:\Program Files\ANSYS Inc\\v201\\ansys\\bin\winx64\ANSYS201.exe", '-b', '-p', 'ANSYS', '-i',
               'D:\Vibration_control_Paper\Ansys_Code\Canonical_Patch\ConicalStructure.txt', '-o',
               'D:\Vibration_control_Paper\Ansys_Code\Canonical_Patch\\ansys_out.txt']
        subprocess.call(cmd)
        print('Y_loc', '|--> '+str(it), 'Z_loc', '|--> '+str(k))
        ## Read The Output Results
        with open('State_Space.csv', 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            i = 0
            Total_vec = np.zeros((4, 12))
            for line in csv_reader:
                Total_vec[i, :] = np.matrix(line)
                i = i + 1

        B_vect = Total_vec[:, 0:6]
        A_vect = Total_vec[:, 6:12]
        B_total[:, 0:6, count] = B_vect
        ## Calculate The fitness function
        LamdaS = 0
        LamdaP = 1
        for N in range(N_Mode):
            LamdaS = LamdaS + np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[0, N])
            LamdaP = LamdaP * np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[0, N])
        Fitness[it, k] = LamdaS * (LamdaP) ** (1 / N_Mode)
        count = count+1

#Pos_vect = Pos_vect[1:-1, :]
#Fitness = Fitness[1:-1, 1:]





Length = 400
Raidus1 = 100
Raidus2 = 0.26*Length + Raidus1
Patch_width = 50
XX = np.zeros((1, 7*17)).T
YY = np.zeros((1, 7*17)).T
i = 0

FF = np.random.rand(1, 7*17).T
for k in range(7):
    for it in range(17):
        XX[i] = k*Patch_width #+ Raidus1
        if it < 9:
            YY[i] = (Raidus2-Raidus1)+it*Patch_width-k*(Patch_width*0.26)
        else:
            YY[i] = (Raidus2 - Raidus1) + it*Patch_width + k * (Patch_width * 0.26)

        plt.plot(XX[i], YY[i], '*')
        i = i + 1


XX1 = XX.reshape((7, 17)).T
YY1 = YY.reshape((7, 17)).T
FF1 = Fitness[1:, 1:]
fig = plt.figure(figsize=(8, 3))
ax = plt.axes(projection='3d')

#clVlu = ['0.1', '0.15', '0.3', '0.55', '0.60', '0.75', '0.80', '0.90', '0.98']
clVlu = ['r', 'b', 'k', 'gray', '0.60', '0.75', '0.80', '0.90', '0.98']
for j in range(7):
    ax.bar3d(XX1[:, j], YY1[:, j], 0*FF1[:, j], 50, 50, FF1[:,  j]/np.max(FF1), clVlu[j])


ax.set_xlabel('Position_X', fontweight='bold')
ax.set_ylabel('Position_Y', fontweight='bold')
ax.set_zlabel('Fitness', fontweight='bold')
ax.xaxis.label.set_size(12)
ax.yaxis.label.set_size(12)
ax.zaxis.label.set_size(10)
plt.grid()
plt.rcParams['font.size'] = '12'

# Save the B matrix results
workbook = xlsxwriter.Workbook('OnePatchBMatrix.xlsx')
worksheet = workbook.add_worksheet()

B_Global = np.zeros((6, 17*7))
for i in range(17*7):
    B_Global[:, i] = B_total[0, 0:6, i]

for col, data in enumerate(list(B_Global[0, :])):
    worksheet.write(0, col, data)
for col, data in enumerate(list(B_Global[1, :])):
    worksheet.write(1, col, data)
for col, data in enumerate(list(B_Global[2, :])):
    worksheet.write(2, col, data)
for col, data in enumerate(list(B_Global[3, :])):
    worksheet.write(3, col, data)
for col, data in enumerate(list(B_Global[4, :])):
    worksheet.write(4, col, data)
for col, data in enumerate(list(B_Global[5, :])):
    worksheet.write(5, col, data)

workbook.close()

A_Matrix = FF1.reshape(1, 7*17)
for i in range(17*7):
    np.savetxt('OnePatchBmatrixConic.txt', B_total[0, 0:6, i])