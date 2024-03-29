import gym
import readchar
import numpy as np
import pickle as pkl
# MACROS
Push_Left = 0
No_Push = 1
Push_Right = 2

# Key mapping
arrow_keys = {
    'A': Push_Left,
    'S': No_Push,
    'D': Push_Right}
end_key = 'Q'
env = gym.make('MountainCar-v0')

end_flag = False
trajectories = []
for episode in range(20): # n_trajectories : 20
    trajectory = []
    env.reset()
    print("episode:{}".format(episode))
    score = 0
    while True: 
        env.render()
        key = readchar.readkey().upper()
        if key not in arrow_keys.keys():
            print('invalid key:{}'.format(key))
            if key == end_key:
                end_flag = True
            break
        action = arrow_keys[key]
        state, reward, done, _ = env.step(action)
        score += reward
        if state[0] >= env.env.goal_position: 
            trajectory.append((state[0], state[1], action))
            env.reset()
            print('mission accomplished! env is reset.')
            break

        trajectory.append((state[0], state[1], action))

    if end_flag:
        print('end!')
        break
    trajectory_numpy = np.array(trajectory, float)
    print("trajectory_numpy.shape", trajectory_numpy.shape)
    print("score:{}".format(score))
    trajectories.append(trajectory)  # don't need to seperate trajectories
env.close()
if not end_flag:
    with open('expert_demo.p',"wb")as f:
        pkl.dump(trajectories,f)
