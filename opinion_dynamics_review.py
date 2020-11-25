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

        delta_op = np.zeros([pop_size, rounds+1])
        delta_op[:,0] = initial_values
        self.delta_op = delta_op



        self.epsilon = np.random.rand(pop_size)*0.3
        #eps = np.random.rand(1)*0.3
        eps = 0.25
        self.epsilon_uniform = eps

        eps_l = 0.1
        eps_r = 0.25
        epsilon_not_sym = np.array([eps_l, eps_r])
        self.epsilon_not_sym = epsilon_not_sym




        self.susceptibility = np.zeros(pop_size)+1





#simple model
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


#bounded confidence model
def bounded_conf(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)

    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = (abs(pop_vector-pop_vector[i])<=epsilon) * pop_vector

        num_of_nonzero = np.count_nonzero(temp)

        new_pop_vector[i] = sum(temp) / num_of_nonzero

    return new_pop_vector


def BC_asymm(pop_vector,pop_size,epsilon):
    new_pop_vector=np.zeros(pop_size)

    for i in range(pop_size):
        #set all values to 0 that are more than epsilon_i apart form x_i
        temp = ((pop_vector-pop_vector[i])<=epsilon[1] )* pop_vector
        temp_1 = (-epsilon[0]<=(pop_vector-pop_vector[i]))*temp

        num_of_nonzero = np.count_nonzero(temp_1)

        new_pop_vector[i] = sum(temp_1) / num_of_nonzero

    return new_pop_vector


confidence_int = 0.5
m = 0.2

def epsilon_pers(pop_vector, pop_size):
    #m = np.random.rand(pop_size)
    epsi = np.ones((pop_size, 2))
    for i in range(pop_size):

        beta_r = m*pop_vector[i] + (1-m)/2
        beta_l = 1 - beta_r

        e_r = beta_r*confidence_int
        e_l = beta_l*confidence_int

        epsi[i,0] = e_l
        epsi[i,1] = e_r

    return epsi




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


def FJ(initial_values, pop_vector, pop_size):
    A = simple_weight(pop_size)
    I = np.eye(pop_size)
    degrees = np.random.rand(pop_size)*0.3
    G = np.diag(degrees)

    new_pop_vector = np.dot(G, initial_values) + np.dot((I - G), np.dot(A, pop_vector))

    return new_pop_vector

def time_variant(initial_values, pop_vector, pop_size, time_step):
    A = simple_weight(pop_size)
    I = np.eye(pop_size)
    time = np.ones(pop_size)*(time_step +1)

    T = np.diag(1/time)

    new_pop_vector = np.dot(T, initial_values) + np.dot((I - T), np.dot(A, pop_vector))

    return new_pop_vector

#####
#initalize
#####

pop = population(500,10) #pop_size, rounds (starting with round 0)

A = simple_weight(pop.pop_size)

#print(pop.opinions, pop.epsilon)



#####
#run
#####


for i in range(pop.rounds):
    #simple weight model
    #pop.opinions[:,i+1] = np.dot(A,pop.opinions[:,i])

    #bounded confidence model with symmetric epsilon
    #pop.opinions[:,i+1] = bounded_conf(pop.opinions[:,i], pop.pop_size, pop.epsilon_uniform)

    #Friedkin and Johnsen model
    #pop.opinions[:,i+1] = FJ(pop.opinions[:, 0], pop.opinions[:, i], pop.pop_size)

    #Time variant model
    #pop.opinions[:,i+1] = time_variant(pop.opinions[:, 0], pop.opinions[:, i], pop.pop_size, i)

    #Bounded confidence model with asymmetric epsilon, uniform epsilon
    #pop.opinions[:,i+1] = BC_asymm(pop.opinions[:,i], pop.pop_size, pop.epsilon_not_sym)

    #Bounded opinon model with asymmetric, non uniform epsilon
    pop.opinions[:,i+1] = BC_asymm_pers(pop.opinions[:,i], pop.pop_size, pop.epsilon_not_sym)

    #Change in opinion
    pop.delta_op[:, i+1] = pop.opinions[:, i+1] - pop.opinions[:, i]
#####
#visualize
#####

#Plot opinion evolution
for i in range(pop.pop_size):
    plt.plot(pop.opinions[i,:],c=plt.get_cmap("jet")(pop.opinions[i,0])) #color according to initial y value
    plt.title("Non-uniform asymmetric boundend confidence $ m = 0.2, \overline{\epsilon} = 0.5$")
    #plt.title("Symmetric uniform confidence interval $\epsilon = 0.25$")

plt.xlabel("Rounds")
plt.ylabel("Opinion")
plt.show()

#Plot change in opinion
'''for i in range(pop.pop_size):
    #plt.plot(pop.delta_op[i,1:],c=plt.get_cmap("jet")(pop.delta_op[i,0])) #color according to initial y value
    #plt.title("Opinion changes for assymmetric non-uniform boundend confidence $m = 0.75, \overline{\epsilon} = 0.4$")
    #plt.title("Opinion changes for symmetric uniform confidence interval $\epsilon = 0.25$")
plt.xlabel("Rounds")
plt.ylabel("Change in opinion")
plt.show()'''
