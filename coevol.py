import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation



def gaussmf(x, sigma=1, scale=1, mean=0):
    return scale * np.exp(-np.square(x - mean) / (2 * sigma ** 2))


k = 4
gamma = 10
phi = 0.1


#number of people
N=100

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

plt.hist(evo_hist, 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)
plt.show()




n = 100
number_of_frames = round(len(evo)/2)
data = evo


def update_hist(num, data):
    plt.cla()
    
    axes = plt.gca()
    axes.set_ylim([0, 50])
    plt.xlabel("Opinions")
    plt.ylabel("Occurences")
    plt.title("t= "+str(num))
    plt.hist(data[num], 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)

fig = plt.figure()
axes = plt.gca()
axes.set_ylim([0, 50])
plt.xlabel("Opinions")
plt.ylabel("Occurences")
plt.title("t= 0")
hist = plt.hist(data[0], 10, range=[0.5, 10.5], edgecolor='black', linewidth=0.7)

animation = animation.FuncAnimation(fig, update_hist, number_of_frames, fargs=(data, ) )
plt.show()
