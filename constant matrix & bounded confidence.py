#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 18 12:11:48 2020

@author: Lukas
"""

import numpy as np
from matplotlib import pyplot as plt



class population:
    def __init__(self, pop_size, rounds):
        
        self.pop_size = pop_size
        self.rounds = rounds
        
        initial_values = np.random.rand(pop_size)
        mat = np.zeros([pop_size,rounds+1])
        mat[:,0] = initial_values
        self.opinions = mat
        
        
        self.epsilon = np.random.rand(pop_size)*0.3
        
        self.susceptibility = np.zeros(pop_size)+1





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


#bounded confidence model
def bounded_conf(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)
    
    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = (abs(pop_vector-pop_vector[i])<=epsilon[i]) * pop_vector
        
        num_of_nonzero = np.count_nonzero(temp)
        
        new_pop_vector[i] = sum(temp) / num_of_nonzero
    
    return new_pop_vector


#####
#initalize
#####

pop = population(60,10) #pop_size, rounds (starting with round 0)

A = simple_weight(pop.pop_size)



#####
#run
#####

for i in range(pop.rounds):
    #simple weight model
    pop.opinions[:,i+1] = np.dot(A,pop.opinions[:,i])
    
    #bounded confidence model
    #pop.opinions[:,i+1] = bounded_conf(pop.opinions[:,i], pop.pop_size, pop.epsilon)


#####
#visualize
#####


for i in range(pop.pop_size):
    plt.plot(pop.opinions[i,:],c=plt.get_cmap("jet")(pop.opinions[i,0])) #color according to initial y value


plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.show()
