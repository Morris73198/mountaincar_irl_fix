import numpy as np
import gym
import random
import sys
import cvxpy as cp

N_idx = 20
F_idx = 4
GAMMA = 0.99

def idx_to_state2(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx
    #position_idx = int((state[0] - env_low[0]) / env_distance[0])
    #velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    position_idx = int((state[0, 0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[0, 1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * N_idx
    return state_idx
    
def idx_to_state0(env, state):
    #print(f"State shape: {state.shape}")
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx
    position_idx = int((state[0] - env_low[0]) / env_distance[0])
    velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    state_idx = position_idx + velocity_idx * N_idx
    return state_idx
    
    
def idx_to_state(env, state):
    env_low = env.observation_space.low
    env_high = env.observation_space.high
    env_distance = (env_high - env_low) / N_idx

    if state.ndim == 1:
        # For 1D state (single observation)
        position_idx = int((state[0] - env_low[0]) / env_distance[0])
        velocity_idx = int((state[1] - env_low[1]) / env_distance[1])
    elif state.ndim == 2:
        # For 2D state (multiple observations)
        position_idx = np.round((state[:, 0] - env_low[0]) / env_distance[0]).astype(int)
        velocity_idx = np.round((state[:, 1] - env_low[1]) / env_distance[1]).astype(int)
    else:
        raise ValueError("Unsupported state dimensions")

    state_idx = position_idx + velocity_idx * N_idx
    return state_idx


if __name__ == '__main__':
    print(":: Testing APP-learning.\n")
    
    # Load the agent
    n_states = N_idx**2  # position - 20, velocity - 20
    n_actions = 3
    q_table = np.load(file="results/app_q_table.npy")

    # Create a new game instance.
    #env = gym.make('MountainCar-v0')
    env = gym.make('MountainCar-v0', render_mode='human')
    #env = gym.make('MountainCar-v0', render_mode='rgb_array')


    n_episode = 10 # test the agent 10times
    scores = []

    
    for ep in range(n_episode):
    	state_tuple = env.reset()
    	state = state_tuple[0]  # Extract the state from the tuple
    	score = 0

    while True:
        # Render the play
        env.render()

        state_idx = idx_to_state(env, state)

        action = np.argmax(q_table[state_idx])

        # Unpack only the necessary values from the returned tuple
        next_state_tuple = env.step(action)
        next_state, reward, done, _ = next_state_tuple[:4]

        next_state = next_state[0] if isinstance(next_state, tuple) else next_state  # Extract the next state from the tuple

        next_state_idx = idx_to_state(env, next_state)

        score += reward
        state = next_state

        if done:
            print('{} episode | score: {:.1f}'.format(ep + 1, score))
            break



    env.close()
    sys.exit()
