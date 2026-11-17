import numpy as np

# States: 0=low traffic, 1=high traffic; actions: 0=keep red, 1=green
Q=np.zeros((2,2)); alpha=.2; gamma=.9; epsilon=.2
rng=np.random.default_rng(1)
for episode in range(1000):
    state=int(rng.integers(0,2))
    action=int(rng.integers(0,2)) if rng.random()<epsilon else int(np.argmax(Q[state]))
    # Reward green; high traffic benefits more from green
    reward = (2 if state==1 else 1) if action==1 else (-2 if state==1 else -1)
    next_state=0 if rng.random()<.7 else 1
    Q[state,action]+=alpha*(reward+gamma*np.max(Q[next_state])-Q[state,action])
print('Learned Q-table:\n',np.round(Q,2))
print('Best action by state:',np.argmax(Q,axis=1),'(1 means green)')
