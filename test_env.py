# Import the environment class from environment.py
from environment import AoIEnvironment

# Create environment object
env = AoIEnvironment()

# Reset environment (start fresh)
state = env.reset()

# Print initial state
print("Initial State:", state)

# Run simulation for 10 steps
for i in range(10):
    
    # Always choose action = 1 (try to transmit)
    action = 1
    
    # Take one step in environment
    next_state, reward, done = env.step(action)
    
    # Print what happened
    print(f"Step {i+1}: State={next_state}, Reward={reward}")