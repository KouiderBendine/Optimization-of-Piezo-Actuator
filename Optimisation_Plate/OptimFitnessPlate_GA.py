import numpy as np
import matplotlib.pyplot as plt
import math
import subprocess
import csv
import os
import random
import time
import xlsxwriter

start_time = time.time()
num_generations = 1000
Patch_N = [2, 3, 4, 5, 6, 7, 8, 9, 10]
NFitnss = np.zeros((9, num_generations+1))
cot = 0
for Patch_Number in Patch_N:
    # -------------------------------------
    Best_Fitness = 0
    X_vec = np.zeros(9)
    for j in range(9):
        X_vec[j] = (j + 1) * 5 + (j) * 50
    Draw = np.zeros((9, 9))
    for k in range(9):
        for it in range(9):
            Draw[it, k] = 1e-3 * X_vec[it]
    Draw_x = Draw.reshape((1, 81))
    Draw_y = Draw.T.reshape((1, 81))
    IterPos = np.concatenate((Draw_y, Draw_x), axis=0)
    # -------------------------------------

    Population_size = int(100)
    qessi = 0.019
    N_Mode = 6
    mutation_rate = 7
    X_vec = np.zeros(9)
    Pos_Accum = np.zeros((Patch_Number, Population_size))
    Vectors = np.zeros((Patch_Number, Population_size))
    best_outputs = np.zeros((Population_size))
    Pos_vect = np.zeros((Patch_Number, Population_size))

    plt_gen = np.ceil(np.linspace(20, 200, 5))


    def Obj_Fun(Posvect, N_Mode, qessi):

        Posvect = np.sort(Posvect)
        ## Filter used to ensure no overlap of the paths
        RAND = np.array([list(range(0, 81))])
        for i in range(Patch_Number):
            for j in range(Patch_Number):
                if j != i:
                    if Posvect[int(j)] == Posvect[int(i)]:
                        arr = np.delete(RAND, np.argwhere(RAND == Posvect[j]))
                        Posvect[j] = np.random.choice(arr)
                        # print(j)
                else:
                    pass

        ## Read The Output Results
        Total_vec = np.loadtxt('OnePatchBMatrix.txt')
        A_vect = np.array([6.561059267, 16.030201499, 40.403947031, 51.245325812, 58.543688261, 102.803257929])

        ## Calculate The fitness function
        LamdaS = 0
        LamdaP = 1
        for N in range(N_Mode):
            LamdaS = LamdaS + np.sum(np.square(Total_vec[N, Posvect.astype(int)])) / (4 * qessi * 2 * np.pi * A_vect[N])
            LamdaP = LamdaP * np.sum(np.square(Total_vec[N, Posvect.astype(int)])) / (4 * qessi * 2 * np.pi * A_vect[N])
        Fitn = LamdaS * (LamdaP) ** (1 / N_Mode)

        return Fitn, Posvect


    def crossover(parent_1, parent_2):
        # Get length of chromosome
        chromosome_length = len(parent_1)
        # Pick crossover point, avoding ends of chromsome
        crossover_point = random.randint(1, chromosome_length - 1)
        # Create children. np.hstack joins two arrays
        child_1 = np.hstack((parent_1[0:crossover_point],
                             parent_2[crossover_point:]))
        child_2 = np.hstack((parent_2[0:crossover_point],
                             parent_1[crossover_point:]))
        # Return children
        return child_1, child_2


    def mutate(population, X_vec, mutation_rate):
        # Apply random mutation
        X_vec
        alf = int(mutation_rate * np.size(population) / 100)

        for i in range(alf):
            random_Row = random.randint(0, np.size(population, axis=0) - 1)
            random_Col = random.randint(0, np.size(population, axis=1) - 1)
            population[random_Row, random_Col] = np.random.choice(81, 1)

        # Return mutation population
        return population


    n_mutations = math.ceil((Population_size - 1) * Patch_Number * mutation_rate)

    ## Creat the starting Population
    for it in range(Population_size):
        Pos_vect[:, it] = np.random.choice(81, Patch_Number)

    ## Fitness initial evaluation
    for iter in range(Population_size):
        Ftn, Posvect = Obj_Fun(Pos_vect[:, iter], N_Mode, qessi)
        Pos_vect[:, iter] = Posvect
        best_outputs[iter] = Ftn
        Ranking = np.argsort(-1 * best_outputs)
        Vectors = np.array(Pos_vect[:, Ranking])
    best_outputs = best_outputs[Ranking]
    best_outputs = best_outputs[0:int(Population_size / 2)]

    Best_Fitness = [best_outputs[0]]
    Best_global = Best_Fitness[0]
    Best_Position = [Vectors[:, 0]]
    Vectors = Vectors[:, 0:int(Population_size / 2)]

    # score = [Best_Fitness]
    #print('Starting best score, percent target: %.10f' % Best_global)
    for generation in range(num_generations):
        #print("Generation : ", generation, "Best Fitness :", Best_Fitness[-1], "Best Fitness :", Best_Position[-1])

        # Create an empty list for new population

        new_population = []

        # Create new popualtion generating two children at a time

        for J in range(int(Population_size / 4)):
            parent_1 = Vectors[:, 2 * J]
            parent_2 = Vectors[:, 2 * J + 1]
            child_1, child_2 = crossover(parent_1, parent_2)
            new_population.append(child_1)
            new_population.append(child_2)

        # Replace the old population with the new one

        population = np.transpose(np.array(new_population))

        # Mutate the population

        population = mutate(population, X_vec, n_mutations)

        # Evaluate the fitness function

        for iter in range(int(Population_size / 2)):
            Ftn, PosVect = Obj_Fun(population[:, iter], N_Mode, qessi)
            population[:, iter] = PosVect
            best_outputs[iter] = Ftn
            Ranking = np.argsort(-1 * best_outputs)
            Vectors = np.array(population[:, Ranking])
        best_outputs = best_outputs[Ranking]
        Best_local = best_outputs[0]

        if Best_local >= Best_global:
            Best_Position.append(Vectors[:, 0])
            Best_Fitness.append(Best_local)
            Best_global = Best_local
        else:
            Best_Position.append(Best_Position[generation])
            Best_Fitness.append(Best_Fitness[generation])

        NFitnss[cot, generation+1] = Best_Fitness[-1]
    cot = cot + 1
        # Normalized  = NFitness-min(NFitness)/ max(NFitness)-min(NFitness)

    # Plot the results
    #plt.figure(1)
    #plt.plot(Best_Fitness, marker='o')
    #plt.xlabel('Generation', fontsize=18, fontweight = 'bold')
    #plt.ylabel('Fitness', fontsize=18, fontweight = 'bold')
    #plt.ylim(0.5e-18, 1.2e-17)
    #plt.grid()
    #plt.show()
    #print(Best_Fitness)
    print('**************')
    print(Best_Position[-1])

    # Save the best results
    workbook = xlsxwriter.Workbook(str('Opt_Posin') + str(Patch_Number) + '.xlsx')

    worksheet = workbook.add_worksheet()
    row = 0
    for col, data in enumerate(Best_Position):
        worksheet.write_column(row, col, data)

    Rw = Patch_Number + 1
    worksheet.write_row(Rw, 0, np.transpose(np.array(Best_Fitness)))

    workbook.close()
# Position
    AL = np.array(Best_Position[900:])
    plt.figure(Patch_Number+1)
    Length = np.linspace(0, 0.5, 10)
    Width = np.linspace(0, 0.5, 10)
    #ax2 = plt.axes()
    #ax2.set_facecolor('gray')
    x_L, y_w = np.meshgrid(Length, Width)
    ax2 = plt.plot(x_L, y_w, color='k', linewidth=4.0)
    ax3 = plt.plot(y_w, x_L, color='k', linewidth=4.0)
    for i in range(np.size(AL, axis=0)):
        A = IterPos[:, AL.astype(int)[i, :]] + np.random.uniform(low=0, high=0.03, size=(2, Patch_Number))
        for j in range(Patch_Number):
            ax1 = plt.plot(A[0, j], A[1, j], color='b', marker='*', linestyle='none')
    plt.savefig('PositionGenPlate' + str(Patch_Number) + '.png')
    plt.show()

plt.figure(12)
for i in range(9):
    plt.plot((NFitnss[i, :]-np.min(NFitnss[i, :]))/(np.max(NFitnss[8, :])-np.min(NFitnss[i, :])))
plt.xlabel('Generation', fontsize=18, fontweight='bold')
plt.ylabel('Normalised Fitness', fontsize=18, fontweight='bold')
plt.legend(['2 Patch', '3 Patch', '4 Patch', '5 Patch', '6 Patch', '7 Patch', '8 Patch', '9 Patch', '10 Patch'])
plt.grid()
plt.show()
plt.ylim(-0.1, 1.1)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
print(time.time() - start_time, "seconds")



#### Save the Fitness Functions
workbook = xlsxwriter.Workbook('FitnessPlateGeneration.xlsx')

worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(NFitnss):
    worksheet.write_column(row, col, data)



workbook.close()




