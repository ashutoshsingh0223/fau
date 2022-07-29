from cartpole import CartPoleEnv
from gridworld import GridWorldEnv

import time


if __name__ == "__main__":
    # Create the environment
    #env = CartPoleEnv()
    env = GridWorldEnv()

    # Reset
    obs = env.reset()
    env.render()

    for i in range(50):
        obs, reward, done, info = env.step(env.action_space.sample())
        env.render()
        # Uncomment this to enable slow motion mode
        #time.sleep(1.0)
        if done:
            env.reset()
    env.close()
