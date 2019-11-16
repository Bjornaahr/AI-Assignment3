#%%
import numpy as np
import pandas as pd
import networkx as nx 
import matplotlib.pyplot as plt
from pprint import pprint 

T = 200 

# %%
hiddenState = ['Happy', 'Sad']
initalState = [0.6, 0.4]

stateSpace = pd.Series(initalState, index=hiddenState, name='states')
print(stateSpace)
print('\n', stateSpace.sum())

# %%
#Transition matrix
tMatrix = pd.DataFrame(columns=hiddenState, index=hiddenState)
tMatrix.loc[hiddenState[0]] = [0.9, 0.2]
tMatrix.loc[hiddenState[1]] = [0.1, 0.8]

print(tMatrix)

a = tMatrix.values
print('\n', a, a.shape, '\n')


# %%
observableState = ['Cooking', 'Crying', 'Sleeping', 'Socializing', 'Watching TV']

b_df = pd.DataFrame(columns=observableState, index=hiddenState)
b_df.loc[hiddenState[0]] = [0.1, 0.2, 0.4, 0.0, 0.3]
b_df.loc[hiddenState[1]] = [0.3, 0.0, 0.3, 0.3, 0.1]

print(b_df)
b = b_df.values
print('\n', b, b.shape, '\n')
print(b_df.sum(axis=1))



# %%
obersvationMap = {'Cooking':0, 'Crying':1, 'Sleeping':2, 'Socializing':3, 'Watching TV':4}
observations = np.random.randint(low=0, high=4, size=T)
#observations = np.array([3, 3, 0, 4, 2])

print(pd.DataFrame(np.column_stack([observations, observationSequence]), columns=['obsCode', 'obsSeq']))

# %%
# define Viterbi algorithm for shortest path
# https://github.com/alexsosn/MarslandMLAlgo/blob/master/Ch16/HMM.py

def viterbi(initalState, a, b, observations):
    
    nStates = np.shape(b)[0]
    T = np.shape(observations)[0]
    # init blank path
    path = np.zeros(T,dtype=int)
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))
    # init delta and phi 
    delta[:, 0] = initalState * b[:, observations[0]]
    phi[:, 0] = 0

    print('\nStart Walk Forward\n')    
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, observations[t]] 
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))
    
    print('-'*50)
    print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    #p('init path\n    t={} path[{}-1]={}\n'.format(T-1, T, path[T-1]))
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        #p(' '*4 + 't={t}, path[{t}+1]={path}, [{t}+1]={i}'.format(t=t, path=path[t+1], i=[t+1]))
        print('path[{}] = {}'.format(t, path[t]))
        
    return path, delta, phi



#%%
path, delta, phi = viterbi(initalState, a, b, observations)
print('\nsingle best state path: \n', path)
print('delta:\n', delta)
print('phi:\n', phi)

#%%
state_map = {0:'Happy', 1:'Sad'}
state_path = [state_map[v] for v in path]


(pd.DataFrame()
 .assign(Observation=observationSequence)
 .assign(Best_Path=state_path))

# %%
np.savetxt("mydata.csv", delta)

# %%
