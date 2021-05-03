import numpy as np
import os
import sys
from datetime import datetime


#%% functions ###

def neighbours(mat, x, y, nx=1, ny=1):
    ''' 
    returns a slice from a lattice (mat) with radius (nx,ny) around 
    coordinate x,y - trimmed by the borders
    ''' 
    return mat[max(x-nx, 0) : min(x+nx+1, mat.shape[0]), \
                      max(y-ny, 0) : min(y+ny+1, mat.shape[1])]



def individualInteraction_Behavior(mySub, neibSub, lattice_ab, params):
    ''' 
    returns the fitness delta of the two interactors according to the cost
    and benefit of the interactions (b and c)
    '''
    c = params['c']
    b = params['b']
    my_ab = lattice_ab[mySub]
    neib_ab = lattice_ab[neibSub]
    
    # coop level: coop only if ab=1 (phenotype a)
    myC   = my_ab * c 
    neibC = neib_ab * c
    
    # fitness calculation according to cooperation
    myfi   = neib_ab * b - myC
    neibfi = my_ab   * b - neibC
    
    return myfi, neibfi



def individualInteraction_Infection(mySub, neibSub, lattice_ab, params):
    ''' 
    the function changes the phenotype lattice (lattice_ab)
    according to horizontal transmission events
    '''
    Ta = params['Ta']
    Tb = params['Tb']
    my_ab   = lattice_ab[mySub]
    neib_ab = lattice_ab[neibSub]
        
    if neib_ab != my_ab:
        if neib_ab == 1: # namely phenotype a
            if np.random.rand() < Ta: # a neib infects with prob Ta
                lattice_ab[mySub] = 1

            if np.random.rand() < Tb:
                lattice_ab[neibSub] = 0

        else: # neib ab is 0, namely phenotype b
            if np.random.rand() < Tb: # b neib infects with prob Tb
                lattice_ab[mySub] = 0

            if np.random.rand() < Ta:
                lattice_ab[neibSub] = 1



def interactions(lattice_ab, params):
    '''
    Performs all the interactions (behavbior and horizontal tranmission) across the
    population in a given generation. The function generates the fitness 
    lattice (lattice_fi) and changes the phenotype lattice (lattice_ab).
    '''
    
    N = params['N']
    nei = params['nei']
    K = params['K']
    InterProb = params['InterProb']
    
    lattice_Ids = np.arange(N*N).reshape(N,N)
    lattice_fi = np.zeros((N,N), dtype=float)
    lattice_NumInter = np.zeros((N,N))   
    
    for it in range(K):
        allSubs = [ (i,j) for i in range(N) for j in range(N) ] 
        np.random.shuffle(allSubs)
        for mySub in allSubs:
            
            if np.random.random() < InterProb: 
                # each individual initiates an interaction with probability InterProb

                i = mySub[0]
                j = mySub[1]
                myId = i*N+j 
                neibIds = neighbours(lattice_Ids,i,j,nei,nei).flatten().tolist()
                neibIds.remove(myId) # remove self from list of neighbor ids
                
                # interaction - behavior:
                neibId = np.random.choice(neibIds) # choose a partner
                neibSub = (neibId//N, neibId%N)
                myfi, neibfi = individualInteraction_Behavior(mySub, neibSub, lattice_ab, params)
                lattice_fi[mySub] += myfi
                lattice_fi[neibSub] += neibfi
                lattice_NumInter[mySub] += 1
                lattice_NumInter[neibSub] += 1         
                
                # interaction - horizontal transmission:
                neibId = np.random.choice(neibIds)  # choose a partner
                neibSub = (neibId//N, neibId%N)
                individualInteraction_Infection(mySub, neibSub, lattice_ab, params)
            
    if params['normalizeByNumInteractions']:
        lattice_NumInter[lattice_NumInter==0] = 1
        lattice_fi /= lattice_NumInter

    lattice_fi += 1
    
    return lattice_fi
    


def selection_prop(lattice_ab, lattice_fi, params):
    ''' 
    Generates the offspring lattice_ab, by placing in each site a replicate 
    from the parent population. The parent is chosen among the neighborhood 
    of the designated site, with probabilities proportional to the parents' 
    fitness. 
    '''
    N = params['N']
    nei = params['nei']
    
    new_ab = np.zeros(lattice_ab.shape)
    
    for i in range(N):
        for j in range(N):
            neib_fi = neighbours(lattice_fi, i, j, nei, nei)
            neib_ab = neighbours(lattice_ab, i, j, nei, nei)
            
            # random pick parent by fitness
            freq_vec = np.reshape(neib_fi, neib_fi.size)
            freq_vec = freq_vec / freq_vec.sum()
            ind = np.argmax(np.random.multinomial(1,freq_vec))

            numCols = neib_fi.shape[1]
            sub = (ind // numCols, ind % numCols)

            curr_phenotype = neib_ab[sub]
            new_ab[i,j] = curr_phenotype
    
    return new_ab
    


def selection_allpop(lattice_ab, lattice_fi, params):
    '''
    Generates the offspring lattice_ab, by placing in each site a replicate 
    from the parent population. The parent is chosen among the entire 
    population, with probabilities proportional to the parents' fitness. 
    '''
    N = params['N']
    wbar = lattice_fi.mean()
    probs = lattice_fi / wbar / N**2
    probs = probs.flatten()
    indices = np.random.choice(np.arange(N**2), size=N**2, replace=True, p=probs)
    
    new_ab = np.zeros(lattice_ab.shape)
    
    for i in range(N):
        for j in range(N):
            num = indices[i*N+j]
            new_ab[i,j] = lattice_ab[num//N, num%N]
    
    return new_ab
            

   
def simulation(lattice_ab, params):
    N = params['N']
    nei = params['nei']
    bound = params['bound']
            
    frac_a = [ float(np.sum(lattice_ab))/(N*N) ] 
    
    ########## MAIN LOOP ##########
    for k in range(bound):   # loop over generations
        
        if k%100 == 0:
            print(k)

        ### perform the interactions ###
        lattice_fi = interactions(lattice_ab, params)
       
        ### perform selection and reproduction (local or global): ###
        ### local:  ###
        #lattice_ab = selection_prop(lattice_ab, lattice_fi, params)
        ### global: ###
        lattice_ab = selection_allpop(lattice_ab, lattice_fi, params)
       
        ### save results ###
        frac_a.append(float(np.sum(lattice_ab))/(N*N))
       
        curr_freqs = np.array([frac_a[-1], 1-frac_a[-1]])
        assert ((curr_freqs >= 0).all())
        assert ((curr_freqs <= 1).all())
        
        ### STOPPING CONDITION ###
        if (curr_freqs==1).any():
            print('Finished due to fixation of one type')
            break
        ###########################################################
    
    return frac_a, k
                

#%% parameters #####

N = 100           # lattice size: N X N
K = 1             # number of initiated interactions chances per individual
InterProb = 0.5   # probability to initiate interaction
nei = 1           # neighbourhood radius. 1 -> 8 neighbors. 2-> 24 neighbors
c = 0.02
b = 1.3
Tb = 0.435
Ta = 0.4

bound = 10**4   # number of generations in simulation
normalizeByNumInteractions = 1


### initialize ab matrix ###
lattice_ab = np.random.randint(2, size=(N, N)) # bacteria


### set params dict ###
params = {
            'N': N,
            'K': K,
            'c': c,
            'b': b,
            'Ta' : Ta,
            'Tb' : Tb,
            'nei': nei,
            'InterProb' : InterProb,
            'bound': bound,
            'normalizeByNumInteractions': normalizeByNumInteractions,
}

print(params)

### run the simulation ###
frac_a, gens = simulation(lattice_ab, params)


#%%

from matplotlib import pyplot as plt
frac_a = np.array(frac_a)
frac_b = 1 - np.array(frac_a)
plt.figure()
plt.plot(frac_a)
plt.plot(frac_b)
plt.legend(['a','b'])