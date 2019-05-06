import pickle
import torch
import numpy as np
import gym
from imitaion_model import Net

task_list = [
    # "Hopper-v2",
    # "Ant-v2",
    # "HalfCheetah-v2",
    # "Humanoid-v2",
    "Reacher-v2",
    # "Walker2d-v2",
]

num_rollouts = 20
render = True


def run(task, train_type):

    print(f"task : {task}")
    conf_dict = {}
    with open(f"{train_type}/{task}.conf", "r", encoding="utf-8") as model_conf:
        for line in model_conf.readlines():
            split_line = line.replace("\n", "").split("=")
            conf_dict[split_line[0]] = int(split_line[1])
    input_size = conf_dict["input_size"]
    hidden_size = conf_dict["hidden_size"]
    output_size = conf_dict["output_size"]
    imitation_net = Net(input_size, hidden_size, output_size)
    imitation_net.load_state_dict(torch.load(f"{train_type}/{task}.pth"))
    imitation_net.eval()

    env = gym.make(task)
    max_steps = env.spec.timestep_limit

    returns = []
    observations = []
    for i in range(num_rollouts):
        print(f"roll out : {i}")
        obs = env.reset()
        done = False
        rewards = 0.
        steps = 0
        while not done:
            observations.append(obs)
            obs_tensor = torch.FloatTensor(obs)
            action = np.ndarray(shape=(1, output_size), buffer=np.array(imitation_net(obs_tensor).data.tolist()))
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(rewards)

    print(f"returns : {returns}")
    print(f"mean return : {np.mean(returns)}")
    print(f"std of return : {np.std(returns)}")

    imitation_data = {"observations": np.array(observations)}

    with open(f"imitations_obs/{task}.pkl", "wb") as f:
        pickle.dump(imitation_data, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    for task_name in task_list:
        run(task_name, "dagger")
