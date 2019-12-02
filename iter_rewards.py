import numpy as np
f = open('HalfCheetah_3.out', 'r')
lines = f.read().splitlines()
f.close()
steps = []
rewards = []
for i in range(len(lines)):
    line = lines[i]
    if line.startswith('Updates'):
        steps.append(int(line.split(',')[1].split(' ')[-1]))
        next_line = lines[i+1]
        reward = next_line.split(',')[0].split(' ')[-1].split('/')[0]
        rewards.append(float(reward))

steps_rewards = np.stack([steps, rewards])
np.save('cheetah_3.npy', steps_rewards)
