import pickle
import tensorflow as tf
import numpy as np
import tf_util
import gym
import load_policy

task_list = [
    # "Hopper-v2",
    # "Ant-v2",
    # "HalfCheetah-v2",
    # "Humanoid-v2",
    # "Reacher-v2",
    "Walker2d-v2"
]

max_timesteps = 5000
num_rollouts = 100
render = False


def main():

    for task in task_list:
        policy_fn = load_policy.load_policy(f"experts/{task}.pkl")

        with tf.Session():
            tf_util.initialize()

            env = gym.make(task)
            max_steps = max_timesteps or env.spec.timestep_limit

            returns = []
            observations = []
            actions = []
            for i in range(num_rollouts):
                print(f"roll out : {i}")
                obs = env.reset()
                done = False
                rewards = 0.
                steps = 0
                while not done:
                    action = policy_fn(obs[None, :])
                    observations.append(obs)
                    actions.append(action)
                    obs, reward, done, _ = env.step(action)
                    rewards += reward
                    steps += 1
                    if render:
                        env.render()
                    if steps % 100 == 0:
                        print(f"{steps} / {max_steps}")
                    if steps >= max_steps:
                        break
                returns.append(rewards)

            print(f"returns : {returns}")
            print(f"mean return : {np.mean(returns)}")
            print(f"std of return : {np.std(returns)}")

            expert_data = {"observations": np.array(observations), "actions": np.array(actions)}

            with open(f"experts_rollout/{task}.pkl", "wb") as f:
                pickle.dump(expert_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
