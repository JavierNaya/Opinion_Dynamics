#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Nov 25 17:22:54 2020

@author: DJ LAV

version: 4
"""
# %% Description

'''
This document contains all the relevant code for our project. It is divided into multiple cells, so make sure you use a program that can detect those and run only one section at a time (e.g. Spyder, with "shift-enter" to run the active section)

The first section contains all general setup and all the functions and classes used in the models. Run this first.

Then there are 4 sections containing the different models used in the project:
    1. classical model: celebrities
    2. bounded confidence, pt. 1
    3. bounded confidence, pt. 2 (using different initial opinion distribution)
    4. YYY

In each of those sections you can find a setup part, this is where you need to specify all the parameters you want to use to run the code. Everything else can be left alone!
To use any model, just run the entire section after choosing the parameters.

If you are using spyder, you may need to change the following setting to have the animation of the coevolution network display correctly:
    Open spyder preferences and then go to IPython Console > Graphics > Backend and change it to "Automatic".
'''



# %% General setup and definition of functions

import numpy as np
import random
import matplotlib.animation as animation
from matplotlib import pyplot as plt
import copy
import scipy.stats as stats


#to prevent savefig from cutting of part of the title
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

#activate to make the graphs bigger
#rcParams['figure.figsize'] = 20, 20



#population class, used for all simulations
class population:
    def __init__(self, pop_size, rounds, distribution="random"):#pop_size, rounds (starting with round 0)
        
        #size of the population
        self.pop_size = pop_size

        #number of rounds
        self.rounds = rounds
        
        #generate initial opinion values using different distributions:
        
        if distribution == "uniform": #(not used)
            #Uniformly spaced opinion 
            initial_values = np.linspace(0.01,1,pop_size)
        
        elif distribution == "random": #standard
            #randomly distributed uniform
            initial_values = np.random.rand(pop_size)
        
        elif distribution == "normal":
            #truncated normal distribution
            lower, upper = 0, 1
            mu, sigma = .5, 0.25
        
            initial_values = stats.truncnorm(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma).rvs(pop_size)
        
        elif distribution == "2normal":
            #two nodes of a normal 
            lower, upper = 0, 1
            mu_l, sigma_l = .2, 0.15
            mu_r, sigma_r = .8, 0.15
            
            initial_values_l = stats.truncnorm(
            (lower - mu_l) / sigma_l, (upper - mu_l) / sigma_l, loc=mu_l, scale=sigma_l).rvs(int(pop_size/2))
            
            initial_values_r = stats.truncnorm(
            (lower - mu_r) / sigma_r, (upper - mu_r) / sigma_r, loc=mu_r, scale=sigma_r).rvs(int(pop_size/2))
            
            initial_values = np.concatenate([initial_values_l, initial_values_r])
        
        
        #create empty opinion matrix and add the initial values
        matrix = np.zeros([pop_size,rounds+1])
        matrix[:,0] = initial_values
        self.opinions = matrix
        
        
#code to visualize the matrix, used for classical model
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


#create community matrix (matrix that represents the connections and is used to calculate the opinions), used for classical model
def comm_matrix(population):
     #for this to work properly, the initial opinion values have to be shuffled randomly, bc this matrix won't be shuffled
     
     #comm_factor: how big different communities are (1 is one large community, 0 is no communities)
     #all communities will have the same size
     #connectedness: how connected people are with different communities (1 = everyone knows everone, 0 = no connections outside community)
     
     #final matrix A is sum of community matrix C and matrix with long-distance connections D (connectednes comes to play here)
     
     if 0<= population.comm_factor <= 1:
         pass
     else:
         print("ERROR, wrong comm_size")
         return 
     
     if 0<= population.connectedness <= 1:
         pass
     else:
         print("ERROR, wrong connectedness")
         return 
     
        
     #Generate C
     #choose how many communities to divide pop into (according to comm_factor)
     population.num_of_comm = population.pop_size * (1-population.comm_factor)
     population.num_of_comm = int(population.num_of_comm)
     population.num_of_comm = max(population.num_of_comm,1)
     population.num_of_comm = min(population.num_of_comm,population.pop_size)
     
     population.comm_size = int(population.pop_size / population.num_of_comm) #size of 1 community
     
     print("#comm: " + str(population.num_of_comm))
     print("size: " + str(population.comm_size))
     
     #initialize matrix C
     C = np.zeros([population.pop_size,population.pop_size])
     
     #populate matrix C
     j=0 
     for i in range(population.num_of_comm):
         #generate random submatrix and copy it onto diagonal of C
         T = np.random.rand(population.comm_size,population.comm_size)
         C[j:j+population.comm_size,j:j+population.comm_size] = T
         j=j+population.comm_size
     

     if j<population.pop_size: 
         #if there are people that didn't fit into communities make them only depending on themselfs
         C[j:,j:] = np.eye(population.pop_size-j)
 
     
 
     #Generate D
     #go through each entry of D and with probability of "connectedness" create entry with random value (also do this for people in own community)

     population.num_of_conn=0 #ceep track of number of connections

     #initialize matrix D
     D = np.zeros([population.pop_size,population.pop_size])
     
     #populate matrix D
     for i in range(population.pop_size):
         for j in range(population.pop_size):
             if i==j: #ignore persons connection to themself
                 continue
             r = np.random.rand(1)[0] #choose random value
             if r <= population.connectedness: #choose if there is a connection in this entry
                 #they know each other
                 D[i,j] = np.random.rand(1)[0]*5 #random connection strenth
                 population.num_of_conn+=1
     
     print("#con: " + str(population.num_of_conn))
     A = C + D
     
     #add more weight on own opinion by adding a multiple of the idenitiy matrix
     A = A + np.eye(population.pop_size)*60
     
     return normalize_mat(A)
     
 
#normalize the matrix, ie make sure all the rows add up to 1, used for classical model
def normalize_mat(A):
     #assuming square matrix
     for i in range(len(A)):
         total = sum(A[i,:])
         
         #if entire row is 0, set the entry on the diagonal to 1 (very unlikely / impossible)
         if total == 0: 
             A[i,i] = 1
         else:
             A[i,:] = A[i,:] / total
     return(A)
 
 
#add celebrities into the matrix by picking a random column (=person) and adding lots of values to it, used for classical model
def add_celebrities(population,num_of_celebs,conn_prob,conn_strength):
     population.conn_prob = conn_prob #probability that a person is connected to any given celebrity
     population.conn_strength = conn_strength #range of how strong a person is connected to any given celebrity it is already connected to (will still get normalized with all the other connections)
     
     #to avoid infinite loop...
     if num_of_celebs > population.pop_size:
         return "ERROR, too many celebtrities"
     population.num_of_celebs = num_of_celebs
     
     
     celeb_list =[] #keep track of all celebrities such that we don't pick the same person twice
     for i in range(population.num_of_celebs):
         cel = np.random.randint(0,population.pop_size)
         while cel in celeb_list: #if already chosen before...
             cel = np.random.randint(0,population.pop_size)
         celeb_list.append(cel)
         
         for row in range(population.pop_size):
             if row == cel: #don't change the celebrities own opinion of him/herself
                 continue
             
             r = np.random.rand(1)[0]
             if r <= population.conn_prob:
                 population.mat[row, cel] = np.random.uniform(population.conn_strength[0], population.conn_strength[1]) #add the connection to the matrix, erasing any previous value
             
     population.mat = normalize_mat(population.mat) #normalize the matrix

                        
#simple weight, used for bounded confidence pt. 1
def simple_weight(pop_size):
    A = np.random.rand(pop_size,pop_size)

    A = (A-0.5).clip(0) #create some 0 values
    A = A + np.eye(pop_size)*6 #more weight on own opinion


    #normalize
    for i in range(pop_size):
        total = sum(A[i,:])

        if total == 0: #unlikely, so idc
            A[i,i] = 1
            print("0 total")
        else:
            A[i,:] = A[i,:] / total
    return(A)


#bounded confidence model, used for bounded confidence pt. 1 & 2
def bounded_conf(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)

    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = (abs(pop_vector-pop_vector[i])<=epsilon) * pop_vector

        num_of_nonzero = np.count_nonzero(temp)

        new_pop_vector[i] = sum(temp) / num_of_nonzero

    return new_pop_vector


#bounded confidence model asymmetric, used for bounded confidence pt. 1
def BC_asymm(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)

    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = ((pop_vector-pop_vector[i])<=epsilon[1] )* pop_vector
        temp_1 = (-epsilon[0]<=(pop_vector-pop_vector[i]))*temp

        num_of_nonzero = np.count_nonzero(temp_1)

        new_pop_vector[i] = sum(temp_1) / num_of_nonzero

    return new_pop_vector


#create personalized epsilon vector, used for bounded confidence pt. 1
def epsilon_pers(pop_vector, pop_size):
    epsi = np.ones((pop_size, 2))
    for i in range(pop_size):

        beta_r = m*pop_vector[i] + (1-m)/2
        beta_l = 1 - beta_r

        e_r = beta_r*confidence_int
        e_l = beta_l*confidence_int

        epsi[i,0] = e_l
        epsi[i,1] = e_r

    return epsi


#bounded confidence model asymmetric with personalized epsilon, used for bounded confidence pt. 1
def BC_asymm_pers(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)
    epsi = epsilon_pers(pop_vector, pop_size)

    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = ((pop_vector-pop_vector[i])<=epsi[i,1] )* pop_vector
        temp_1 = (-epsi[i,0]<=(pop_vector-pop_vector[i]))*temp

        num_of_nonzero = np.count_nonzero(temp_1)

        new_pop_vector[i] = sum(temp_1) / num_of_nonzero

    return new_pop_vector


#time variant model, used for bounded confidence pt. 1
def time_variant(initial_values, pop_vector, pop_size, time_step):
    A = simple_weight(pop_size)
    I = np.eye(pop_size)
    time = np.ones(pop_size)*(time_step +1)

    T = np.diag(1/time)

    new_pop_vector = np.dot(T, initial_values) + np.dot((I - T), np.dot(A, pop_vector))

    return new_pop_vector


#bifurcation diagram, used for bounded confidence pt. 2
def bifurcation(pop_size,num_epsilon,rounds,distribution = 'uniform'):
    j = 0 #count of epsilon round 
    final_count = np.zeros([num_epsilon])
    final_values = [] #List with the final values obtained

    for eps in np.linspace(0.01,1,num_epsilon): #scan the space of epsilon
        
        pop = population(pop_size,rounds,distribution = distribution) #pop_size, rounds (starting with round 0)
        pop.epsilon = eps*np.ones([pop_size])

        for i in range(pop.rounds):
            #bounded confidence model            
            pop.opinions[:,i+1] = bounded_conf(pop.opinions[:,i], pop.pop_size, pop.epsilon)
            
            
        #count number of unique opinions 
        final_values.append(np.unique(pop.opinions[:,-1]))
        final_count[j] = len(final_values[j])
        j += 1
        print(j/num_epsilon)

    return final_values





def gaussmf(x, sigma=1, scale=1, mean=0): #used for coevolution
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))


def update_hist(num, data): #used for coevolution
    plt.cla()
    
    axes = plt.gca()
    axes.set_ylim([0, 50])
    plt.xlabel("Opinions")
    plt.ylabel("Occurences")
    plt.title("t= "+str(num))
    plt.hist(data[num], 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)



# %% 
########################
# Model 1: classical model: celebrities
########################

#####
# setup
#####

#specify number of experiment (used to name the saved graphs)
number = "01" 

#used 80 in the paper
population_size = 80

#(starting with round 0), used 50 in the paper
number_of_rounds = 50 

#how big different commuities are (1 is one large community, 0 is no communities), used 0.93 in the paper
comm_factor = 0.93 

#how connected people are with people from different communities (1 = everyone knows everone, 0 = no connections outside community), used 0 and 0.93 in the paper
connectedness = 0 

#number of celebrities, used 1, 3, 4 in the paper
num_of_celebs = 3

#probability that a person is connected to any given celebrity, used 0.3 in the paper
conn_prob = 0.3

#range of how strong a person is connected to any given celebrity it is already connected to, used [0,0.2] in the paper
conn_strength = [0, 0.2]




#####
# initalize
#####

#create a population with certain parameters
control_pop = population(population_size, number_of_rounds) #population size, rounds (starting with round 0)

#add additional parameters:

#comm_factor: how big different commuities are (1 is one large community, 0 is no communities) 
control_pop.comm_factor = comm_factor

#connectedness: how connected people are with people from different communities (1 = everyone knows everone, 0 = no connections outside community)
#(in the paper we used 0 for "basic" and 0.93 "for more connections" )
control_pop.connectedness = connectedness 

#create matrix with communities
control_pop.mat = comm_matrix(control_pop)

#keep track of number of celebrities, the probability the are connected to a given agent and the interval of possible connection strength
control_pop.num_of_celebs = 0 #don't change
control_pop.conn_prob = 0 #don't change
control_pop.conn_strength = [0,0] #don't change


#copy the class to create a duplicate populaiton
celeb_pop = copy.deepcopy(control_pop)

#add celebrities to the duplicate population
# in the paper we used (x, 0.3, [0,0.2]) with x = 1, 3, 4
add_celebrities(celeb_pop, num_of_celebs, conn_prob, conn_strength) 


#####
# run
#####

#control
for i in range(control_pop.rounds):
    control_pop.opinions[:,i+1] = np.dot(control_pop.mat,control_pop.opinions[:,i])
    
    
#celeb
for i in range(celeb_pop.rounds):
    celeb_pop.opinions[:,i+1] = np.dot(celeb_pop.mat,celeb_pop.opinions[:,i])




#####
# visualize
#####

#create graphs for control
plt.figure(1)
for i in range(control_pop.pop_size):
    plt.plot(control_pop.opinions[i,:],c=plt.get_cmap("jet")(control_pop.opinions[i,0])) #color according to initial y value


plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.title("Control Population\n pop_size="+ str(control_pop.pop_size) + " comm_factor=" + str(control_pop.comm_factor) + "\n #comm: " + str(control_pop.num_of_comm) + " comm_size: " + str(control_pop.comm_size) + " connectedness:" + str(control_pop.connectedness) + "\n #conn: " + str(control_pop.num_of_conn) + " #celebs: " + str(control_pop.num_of_celebs) + "\n conn_prob: " + str(control_pop.conn_prob) + " conn_strength: " + str(control_pop.conn_strength))
plt.show()
#save png
#plt.savefig("celeb_" + number + ("_control_opinions.png"))

plt.figure(2)
hinton(np.matrix.transpose(control_pop.mat))
plt.show
#save png
#plt.savefig("celeb_" + number + ("_control_matrix.png"))


#create graphs for celebrity matrix
plt.figure(3)
for i in range(celeb_pop.pop_size):
    plt.plot(celeb_pop.opinions[i,:],c=plt.get_cmap("jet")(celeb_pop.opinions[i,0])) #color according to initial y value

plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.title("Celebrity Population\n pop_size="+ str(celeb_pop.pop_size) + " comm_factor=" + str(celeb_pop.comm_factor) + "\n #comm: " + str(celeb_pop.num_of_comm) + " comm_size: " + str(celeb_pop.comm_size) + " connectedness:" + str(celeb_pop.connectedness) + "\n #conn: " + str(celeb_pop.num_of_conn)+ " #celebs: " + str(celeb_pop.num_of_celebs) + "\n conn_prob: " + str(celeb_pop.conn_prob) + " conn_strength: " + str(celeb_pop.conn_strength))
plt.show()
#save png
#plt.savefig("celeb_" + number + ("_celeb_opinions.png"))

plt.figure(4)
hinton(np.matrix.transpose(celeb_pop.mat))
plt.show()
#save png
#plt.savefig("celeb_" + number + ("_celeb_matrix.png"))


#####
# save matrix data to be able to retrieve later
#####

# #save the data as a npy file to be able to retrieve the data later to redraw graphs in a different way
# with open('celeb_' + number + '.npy','wb') as f:
#     np.save(f,control_pop.mat)
#     np.save(f,control_pop.opinions)
#     np.save(f,celeb_pop.mat)
#     np.save(f,celeb_pop.opinions)



#to retrieve:
# with open('celeb_0.npy','rb') as f:
#     a=np.load(f) #control_pop.mat
#     b=np.load(f) #control_pop.opinions
#     c=np.load(f) #celeb_pop.mat
#     d=np.load(f) #celeb_pop.opinions

print("DONE")


# %% 
########################
# Model 2: bounded confidence, pt. 1
########################

#####
# setup
#####

#choose 1 of the 4 models:
#model = "bounded confidence"  #bounded confidence model with symmetric epsilon
#model = "time variant"  #Time variant model
model = "bounded confidence, asymmetric"  #Bounded confidence model with asymmetric epsilon, uniform epsilon
#model = "bounded confidence, asymmetric, non uniform"   #Bounded opinon model with asymmetric, non uniform epsilon

population_size = 500

#starting with round 0
number_of_rounds = 10

#set uniform epsilon
epsilon_uniform = 0.25

#set bound for asymmterical epsilon values
eps_l = 0.1 #left bound
eps_r = 0.25 #right bound

#set confidence interval (used 0.5)
confidence_int = 0.5

#set m (used 0.2)
m = 0.2


#####
# initalize
#####

#create population
pop = population(population_size,number_of_rounds) #pop_size, rounds (starting with round 0)


#add additional parameters:

#create matrix to calculate the opinion-deltas
delta_op = np.zeros([pop.pop_size, pop.rounds+1])
delta_op[:,0] = pop.opinions[:,0]
pop.delta_op = delta_op

#randomly choose standard epsilon
pop.epsilon = np.random.rand(pop.pop_size)*0.3

#set uniform epsilon
pop.epsilon_uniform = epsilon_uniform

#set asymmetrical epsilon values
eps_l = 0.1 #left bound
eps_r = 0.25 #right bound
epsilon_not_sym = np.array([eps_l, eps_r])
pop.epsilon_not_sym = epsilon_not_sym


#####
# run
#####

for i in range(pop.rounds):

    
    if model == "bounded confidence":
        #bounded confidence model with symmetric epsilon
        pop.opinions[:,i+1] = bounded_conf(pop.opinions[:,i], pop.pop_size, pop.epsilon_uniform)
        
    elif model == "time variant":
        #Time variant model
        pop.opinions[:,i+1] = time_variant(pop.opinions[:, 0], pop.opinions[:, i], pop.pop_size, i)

    elif model == "bounded confidence, asymmetric":
        #Bounded confidence model with asymmetric epsilon, uniform epsilon
        pop.opinions[:,i+1] = BC_asymm(pop.opinions[:,i], pop.pop_size, pop.epsilon_not_sym)
        
    elif model == "bounded confidence, asymmetric, non uniform":
        #Bounded opinon model with asymmetric, non uniform epsilon
        pop.opinions[:,i+1] = BC_asymm_pers(pop.opinions[:,i], pop.pop_size, pop.epsilon_not_sym)


    
    #calculate change in opinion
    pop.delta_op[:, i+1] = pop.opinions[:, i+1] - pop.opinions[:, i]


#####
# visualize
#####


#Plot opinion evolution
for i in range(pop.pop_size):
    plt.plot(pop.opinions[i,:],c=plt.get_cmap("jet")(pop.opinions[i,0])) #color according to initial y value
    
    if model == "bounded confidence":
        plt.title("Symmetric uniform confidence interval $\epsilon = " + str(pop.epsilon_uniform) + "$")
        
    elif model == "time variant":
        plt.title("time variant model")
    
    elif model == "bounded confidence, asymmetric":
        plt.title("Asymmetric uniform confidence interval $\epsilon_l = " + str(eps_l) + "\epsilon_l = " + str(eps_r) + "$")
        
    elif model == "bounded confidence, asymmetric, non uniform":
        plt.title("Non-uniform asymmetric boundend confidence $ m = " + str(m) + ", \epsilon_l = " + str(eps_l) + "\epsilon_l = " + str(eps_r)  + "$")


plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.show()

#Plot change in opinion
for i in range(pop.pop_size):
    plt.plot(pop.delta_op[i,1:],c=plt.get_cmap("jet")(pop.delta_op[i,0])) #color according to initial y value
    
    if model == "bounded confidence":
        plt.title("Opinion changes for symmetric uniform confidence interval $\epsilon = " + str(pop.epsilon_uniform) + "$")
        
    elif model == "time variant":
        plt.title("Opinion changes for time variant model")
        
    elif model == "bounded confidence, asymmetric":
        plt.title("Opinion changes for asymmetric uniform bounded confidence $\epsilon = " + str(eps_l) + "\epsilon_l = " + str(eps_r) + "$")
        
    elif model == "bounded confidence, asymmetric, non uniform":
        plt.title("Opinion changes for asymmetric non-uniform boundend confidence $m = " + str(m) + ", \epsilon_l = " + str(eps_l) + "\epsilon_l = " + str(eps_r)  + "$")

plt.xlabel("Rounds")
plt.ylabel("Change in opinion")
plt.show()


print("DONE")


# %% 
########################
# Model 2: bounded confidence, pt. 2
########################

#####
# setup
#####

#choose 1 of the 4 distributions
#distribution = "normal"  #truncated normal distribution
#distribution = "2normal"  #two nodes of a normal
#distribution = "random"  #randomly distributed uniform  
distribution = "uniform"  #Uniformly spaced opinion 

population_size = 200

#starting with round 0
number_of_rounds = 100

#set number of steps for epsilon
num_epsilon = 100
 

#####
# run
#####

#Bifurcation run 
final_values = bifurcation(population_size,num_epsilon,number_of_rounds,distribution)


#####
# visualize
#####

#Bifurcation visualization 
i = 0
x, y = [], []
#prepare data for plotting
for eps in np.linspace(0.01,1,num_epsilon):
    y.extend(final_values[i])
    x.extend(eps*np.ones(len(final_values[i])))
    i += 1
    

plt.scatter(x,y,c=y,cmap='jet',s=10)
plt.xlabel("$\epsilon$", size=26)
plt.ylabel("Opinion", size=26)
#plt.title('Bifurcation Diagram - Population' + str(pop_size) )
plt.savefig('Symmetric Trial - ' + distribution + '.png')




# %% 
########################
# Model 3: coevolution network
########################

#####
# setup
#####
k = 4
gamma = 10
phi = 0.1

#number of people
N=100

#####
# initalize & run
#####
M= round(N*k/2)
G = round(N/gamma)

#opinions of people
x = [random.randint(1, G) for i in range(N)]
m = np.zeros((N, N))


l = M
while l>0:
    i = random.randint(0, N-1)
    j = random.randint(0, N-1)
    if m[i][j]==0 and i!=j:
        m[i][j]=1
        m[j][i]=1
    l=l-1

evo = [list(x)]
t_end = 1000
t=0

while t<=t_end:
    last_x = evo[-1]
    i = random.randint(0, N-1)

    if sum(m[i]) != 0: #do nothing if i has no edges 
        h = [] # same opinions
        f=0

        for f in range(N):
            if x[f] == x[i] and f!=i:
                h.append(f)

        c=[] #other end of vertices
        r=0
        for r in range(N):
            if m[r][i]==1:
                c.append(r)
        
        if random.random() <= phi and h:
            q=random.randint(0, len(c)-1)
            hh=h
            s=0
            while s==0 and sum(hh)>0:
                j=random.randint(0, len(hh)-1)
                while hh[j]==0:
                    j=random.randint(0, len(hh)-1)
                if m[hh[j]][i]==0:
                    s=1
                else:
                    hh[j]=0

            m[i][c[q]] =0
            m[c[q]][i]=0
            m[i][h[j]]=1
            m[h[j]][i]=1
        else:
            s=0
            while s==0:
                j=random.randint(0, len(c)-1)
                a=random.random()
                b=random.random()
                if evo[0][i]==x[c[j]]:
                    s=1 
                    x[i]=x[c[j]]
                elif a <= 1.666*abs(1/(evo[0][i]-x[c[j]])) and b <= gaussmf(sum([l[c[j]] for l in m]), M*0.3, M/2):
                    s=1
                    x[i]=x[c[j]]
                else:
                    x[i]=x[c[j]]
            
    else:
        t=t-1
    
    evo.append(x.copy())
    t=t+1


evo_hist = evo[-1]

#####
# visualize
#####


plt.figure()
plt.hist(evo_hist, 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)
plt.xlabel("Opinions")
plt.ylabel("Occurences")
plt.title("t= "+str(len(evo)-1))
plt.show()


fig = plt.figure()
axes = plt.gca()
axes.set_ylim([0, 50])
plt.xlabel("Opinions")
plt.ylabel("Occurences")
plt.title("t= 0")
hist = plt.hist(evo[0], 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)


animation = animation.FuncAnimation(fig, update_hist, fargs=(evo, ) )
plt.show()
