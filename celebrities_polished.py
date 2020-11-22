#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 16:51:28 2020

@author: Lukas
"""

###
# to run, make changes in the "Initialize" part
###

import numpy as np
from matplotlib import pyplot as plt
import copy


#to prevent savefig from cutting of part of the title
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#to set size of exportet plots
rcParams['figure.figsize'] = 20, 20


#code to visualize the matrix,
def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix."""
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()
    



#population class
class population:
    def __init__(self, pop_size, rounds, comm_factor, connectedness):#pop_size, rounds (starting with round 0), comm_factor, connectedness
        
        #size of the population
        self.pop_size = pop_size

        #number of rounds
        self.rounds = rounds
        
        #generate initial values, either randomized or not ramdomized
        initial_values = np.random.rand(pop_size) #initial opinions randomized
        #initial_values = np.linspace(0,1,pop_size) #not randomized
        
        #create opinion matrix
        mat = np.zeros([pop_size,rounds+1])
        mat[:,0] = initial_values
        self.opinions = mat
        
        #parameter to set community size
        self.comm_factor = comm_factor
        
        #parameter to set how many connections there are between different communities
        self.connectedness = connectedness
        
        #create matrix with communities
        self.mat = self.comm_matrix()
        
        #keep track of number of celebrities, the probability the are connected to a given agent and the interval of possible connection strength
        self.num_of_celebs = 0
        self.conn_prob = 0
        self.conn_strength = [0,0]
        
        
        
    
    def comm_matrix(self):
        #for this to workproperly, the initial opinion values have to be shuffled randomly, bc this matrix won't be shuffled
        
        #comm_factor: how big different communities are (1 is one large community, 0 is no communities)
        #all communities will have the same size
        #connectedness: how connected people are with different communities (1 = everyone knows everone, 0 = no connections outside community)
        
        #final matrix A is sum of community matrix C and matrix with long-distance connections D
        
        if 0<= self.comm_factor <= 1:
            pass
        else:
            print("ERROR, wrong self.comm_size")
            return 
        
        if 0<= self.connectedness <= 1:
            pass
        else:
            print("ERROR, wrong connectedness")
            return 
        
        #Generate C
        #choose how many communities to divide pop into
        self.num_of_comm = self.pop_size * (1-self.comm_factor)
        self.num_of_comm = int(self.num_of_comm)
        self.num_of_comm = max(self.num_of_comm,1)
        self.num_of_comm = min(self.num_of_comm,self.pop_size)
        
        self.comm_size = int(self.pop_size / self.num_of_comm) #size of 1 community
        
        print("#comm: " + str(self.num_of_comm))
        print("size: " + str(self.comm_size))
        
        #initialize matrix C
        C = np.zeros([self.pop_size,self.pop_size])
        
        #populate matrix C
        j=0 
        for i in range(self.num_of_comm):
            #generate random submatrix and copy it onto diagonal of C
            T = np.random.rand(self.comm_size,self.comm_size)
            C[j:j+self.comm_size,j:j+self.comm_size] = T
            j=j+self.comm_size
        

        if j<self.pop_size: 
            #if there are people that didn't fit into communities make them only depending on themselfs
            C[j:,j:] = np.eye(self.pop_size-j)
    
        
    
        #Generate D
        #go through each entry of D and with probability of "connectedness" create entry with random value (also do this for people in own community)

        self.num_of_conn=0 #ceep track of number of connections

        #initialize matrix D
        D = np.zeros([self.pop_size,self.pop_size])
        
        #populate matrix D
        for i in range(self.pop_size):
            for j in range(self.pop_size):
                if i==j: #ignore persons connection to themself
                    continue
                r = np.random.rand(1)[0] #choose random value
                if r <= self.connectedness: #choose if there is a connection in this entry
                    #they know each other
                    D[i,j] = np.random.rand(1)[0]*5 #random connection strenth
                    self.num_of_conn+=1
        
        print("#con: " + str(self.num_of_conn))
        A = C + D
        
        #add more weight on own opinion by adding a multiple of the idenitiy matrix
        A = A + np.eye(self.pop_size)*60
        
        return self.normalize_mat(A)
        
    
    #normalize the matrix, ie make sure all the rows add up to 1
    def normalize_mat(self,A):
        #assuming square matrix
        for i in range(len(A)):
            total = sum(A[i,:])
            
            #if entire row is 0, set the entry on the diagonal to 1 (very unlikely / impossible)
            if total == 0: 
                A[i,i] = 1
            else:
                A[i,:] = A[i,:] / total
        return(A)
    
    
    #add celebrities into the matrix by picking a random column (=person) and adding lots of values to it
    def add_celebrities(self,num_of_celebs,conn_prob,conn_strength):
        self.conn_prob = conn_prob #probability that a person is connected to any given celebrity
        self.conn_strength = conn_strength #range of how strong a person is connected to any given celebrity it is already connected to (will still get normalized with all the other connections)
        
        if num_of_celebs > self.pop_size:
            return "TOO MANY CELEBRITIES"
        self.num_of_celebs = num_of_celebs
        
        
        celeb_list =[] #keep track of all celebrities such that we don't pick the same person twice
        for i in range(self.num_of_celebs):
            cel = np.random.randint(0,self.pop_size)
            while cel in celeb_list: #if already chosen before...
                cel = np.random.randint(0,self.pop_size)
            celeb_list.append(cel)
            
            for row in range(self.pop_size):
                if row == cel: #don't change the celebrities own opinion of him/herself
                    continue
                
                r = np.random.rand(1)[0]
                if r <= self.conn_prob:
                    self.mat[row, cel] = np.random.uniform(self.conn_strength[0], self.conn_strength[1]) #add the connection to the matrix, erasing any previous value
                
        self.mat = self.normalize_mat(self.mat) #normalize the matrix
        
                        

#####
#initalize
#####

#specify number of experiment (used to name saved graphs)
number = "01" 

#create a population with certain parameters
control_pop = population(80,50,0.93,0.012) #pop_size, rounds (starting with round 0), comm_factor, connectedness

#copy the class to create a duplicate populaiton
celeb_pop = copy.deepcopy(control_pop)

#add celebrities to the duplicate populaiton
celeb_pop.add_celebrities(3, 0.3, [0,0.2]) #num_of_celebs, conn_prob, conn_strength


#####
#run
#####


#control
for i in range(control_pop.rounds):
    control_pop.opinions[:,i+1] = np.dot(control_pop.mat,control_pop.opinions[:,i])
    
    
#celeb
for i in range(celeb_pop.rounds):
    celeb_pop.opinions[:,i+1] = np.dot(celeb_pop.mat,celeb_pop.opinions[:,i])



#####
#visualize
#####

#control
plt.figure(1)
for i in range(control_pop.pop_size):
    plt.plot(control_pop.opinions[i,:],c=plt.get_cmap("jet")(control_pop.opinions[i,0])) #color according to initial y value


plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.title("Control Population\n pop_size="+ str(control_pop.pop_size) + " comm_factor=" + str(control_pop.comm_factor) + "\n #comm: " + str(control_pop.num_of_comm) + " comm_size: " + str(control_pop.comm_size) + " connectedness:" + str(control_pop.connectedness) + "\n #conn: " + str(control_pop.num_of_conn) + " #celebs: " + str(control_pop.num_of_celebs) + "\n conn_prob: " + str(control_pop.conn_prob) + " conn_strength: " + str(control_pop.conn_strength))
#plt.show()
plt.savefig("celeb_" + number + ("_control_opinions.png"))
plt.figure(2)
hinton(np.matrix.transpose(control_pop.mat))
plt.savefig("celeb_" + number + ("_control_matrix.png"))


#celeb
plt.figure(3)
for i in range(celeb_pop.pop_size):
    plt.plot(celeb_pop.opinions[i,:],c=plt.get_cmap("jet")(celeb_pop.opinions[i,0])) #color according to initial y value

plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.title("Celebrity Population\n pop_size="+ str(celeb_pop.pop_size) + " comm_factor=" + str(celeb_pop.comm_factor) + "\n #comm: " + str(celeb_pop.num_of_comm) + " comm_size: " + str(celeb_pop.comm_size) + " connectedness:" + str(celeb_pop.connectedness) + "\n #conn: " + str(celeb_pop.num_of_conn)+ " #celebs: " + str(celeb_pop.num_of_celebs) + "\n conn_prob: " + str(celeb_pop.conn_prob) + " conn_strength: " + str(celeb_pop.conn_strength))
#plt.show()
plt.savefig("celeb_" + number + ("_celeb_opinions.png"))

plt.figure(4)
hinton(np.matrix.transpose(celeb_pop.mat))
plt.savefig("celeb_" + number + ("_celeb_matrix.png"))


#####
#save matrix
#####
#save the data as a npy file to be able to retrieve the data later to redraw graphs in a different way
with open('celeb_' + number + '.npy','wb') as f:
    np.save(f,control_pop.mat)
    np.save(f,control_pop.opinions)
    np.save(f,celeb_pop.mat)
    np.save(f,celeb_pop.opinions)



#to retrieve:
# with open('celeb_0.npy','rb') as f:
#     a=np.load(f) #control_pop.mat
#     b=np.load(f) #control_pop.opinions
#     c=np.load(f) #celeb_pop.mat
#     d=np.load(f) #celeb_pop.opinions

print("DONE")
