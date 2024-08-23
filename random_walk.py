import numpy as np
import matplotlib.pyplot as plt

# Function to generate a random walk
def random_walk(n_steps):
    steps = np.random.choice([-1, 1], size=n_steps)
    
    random_walk = np.cumsum(steps)
    
    return random_walk

n_steps = 50 

walk = random_walk(n_steps)

# Plot the random walk
plt.figure(figsize=(10, 6))
plt.plot(walk, label='Random Walk')
plt.title(f'Random Walk with {n_steps} Steps')
plt.xlabel('Steps')
plt.ylabel('Position')
plt.grid(True)
plt.legend()
plt.show()
