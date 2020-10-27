#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:51:28 2020

@author: Lukas
"""

import numpy as np
from matplotlib import pyplot as plt



class population:
    def __init__(self, pop_size, rounds, comm_factor, connectedness, randomness):#pop_size, rounds (starting with round 0), comm_factor, connectedness, randomness
        
        self.pop_size = pop_size
        self.rounds = rounds
        
        #initial_values = np.random.rand(pop_size) #initial opinions randomized
        initial_values = np.linspace(0,1,pop_size) #not randomized
        
        mat = np.zeros([pop_size,rounds+1])
        mat[:,0] = initial_values
        self.opinions = mat
        
        self.comm_factor = comm_factor
        self.connectedness = connectedness
        self.randomness = randomness
        
        
        
    
def comm_matrix(pop_size, comm_factor, connectedness, randomness):
    #for this to workproperly, the initial opinion values have to be shuffled randomly, bc this matrix won't be shuffled
    
    #comm_factor: how big different communities are (1 is one large community, 0 is no communities)
    #all communities will have the same size (this could be changed)
    #connectedness: how connected people are with different communities (1 = everyone knows everone, 0 = no connections outside community)
    #randomness: randomize comm_factor
    
    #final matrix A is sum of community matrix C and matrix with long-distance connections D
    
    if 0<= comm_factor <= 1:
        pass
    else:
        print("ERROR, wrong comm_size")
        return 
    
    if 0<= connectedness <= 1:
        pass
    else:
        print("ERROR, wrong connectedness")
        return 
    
    #Generate C
    #choose how many communities to divide pop into (in a linear way with some random noise) (I did not really use the randomness yet, don't know if it is useful)
    num_of_comm = pop_size * (1-comm_factor)
    num_of_comm = num_of_comm + 10*(np.random.rand(1)[0]-0.5) * randomness 
    num_of_comm = int(num_of_comm)
    num_of_comm = max(num_of_comm,1)
    num_of_comm = min(num_of_comm,pop_size)
    
    comm_size = int(pop_size / num_of_comm) #size of 1 community
    
    print("#comm: " + str(num_of_comm))
    print("size: " + str(comm_size))
    C = np.zeros([pop_size,pop_size])
    
    j=0 
    for i in range(num_of_comm):
        #generate random submatrix and copy it onto diagonal of C
        T = np.random.rand(comm_size,comm_size)
        C[j:j+comm_size,j:j+comm_size] = T
        j=j+comm_size
    
    if j<pop_size: 
        #if there are people that didn't fit into communities make them only depending on themselfs (might change to create mini community out of those)
        C[j:,j:] = np.eye(pop_size-j)

    

    #Generate D
    #go through each entry of D and with probability of "connectedness" create entry with random value (same as the one used inside community, but could also be scaled by connectedness for example)
    #also do this for people in own community bc otherwise it would be more complicated (:)) and also this is also there is an argument why this might be more realistic that way
    #probably could be solved more efficiently?
    
    c=0 #count number of connections
    D = np.zeros([pop_size,pop_size])
    for i in range(pop_size):
        for j in range(pop_size):
            if i==j: #person connection to themself
                continue
            r = np.random.rand(1)[0]
            if r <= connectedness:
                #they know each other
                D[i,j] = np.random.rand(1)[0]*5
                c+=1
    
    print("#con: " + str(c))
    A = C + D
    
    
    A = A + np.eye(pop_size)*10 #more weight on own optinion
    
    return normalize_mat(A)
        
    





#simple model
def simple_weight(pop_size):
    A = np.random.rand(pop_size,pop_size)
    
    A = (A-0.5).clip(0) #create some 0 values
    A = A + np.eye(pop_size)*6 #more weight on own optinion
   
    
    #normalize
    for i in range(pop_size):
        total = sum(A[i,:])
        
        if total == 0: #unlikely, so idc
            A[i,i] = 1
            print("0 total")
        else:
            A[i,:] = A[i,:] / total
    return(A)



def normalize_mat(A):
    #normalize
    #assuming square matrix
    
    for i in range(len(A)):
        total = sum(A[i,:])
        
        #if entire row is 0, set the entry on the diagonal to 1 (very unlikely / impossible)
        if total == 0: 
            A[i,i] = 1
        else:
            A[i,:] = A[i,:] / total
    return(A)


#####
#initalize
#####

pop = population(80,50,0.9,0.005,0) #pop_size, rounds (starting with round 0), comm_factor, connectedness, randomness 

A = comm_matrix(pop.pop_size, pop.comm_factor, pop.connectedness, pop.randomness) 


#####
#run
#####

for i in range(pop.rounds):
    #simple weight model
    pop.opinions[:,i+1] = np.dot(A,pop.opinions[:,i])
    
#####
#visualize
#####


for i in range(pop.pop_size):
    plt.plot(pop.opinions[i,:],c=plt.get_cmap("jet")(pop.opinions[i,0])) #color according to initial y value


plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.title("constant_,matrix_w_communities\n pop_size="+ str(pop.pop_size) + " comm_factor=" + str(pop.comm_factor) + " connectedness:" + str(pop.connectedness) + " randomness:" + str(pop.randomness))
plt.show()
