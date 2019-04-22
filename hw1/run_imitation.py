import torch
import numpy as np
import gym
from imitaion_model import Net

task_list = [
    "Hopper-v2",
    "Ant-v2",
    "HalfCheetah-v2",
    "Humanoid-v2",
    "Reacher-v2",
    "Walker2d-v2"
]

max_timesteps = 5000
num_rollouts = 5
render = False


def main():

    for task in task_list:
        print(f"task : {task}")
        conf_dict = {}
        with open(f"imitations/{task}.conf", "r", encoding="utf-8") as model_conf:
            for line in model_conf.readlines():
                split_line = line.replace("\n", "").split("=")
                conf_dict[split_line[0]] = int(split_line[1])
        input_size = conf_dict["input_size"]
        hidden_size = conf_dict["hidden_size"]
        output_size = conf_dict["output_size"]
        imitation_net = Net(input_size, hidden_size, output_size)
        imitation_net.load_state_dict(torch.load(f"imitations/{task}.pth"))
        imitation_net.eval()

        env = gym.make(task)
        max_steps = max_timesteps or env.spec.timestep_limit

        returns = []
        for i in range(num_rollouts):
            print(f"roll out : {i}")
            obs = env.reset()
            done = False
            rewards = 0.
            steps = 0
            while not done:
                obs_tensor = torch.FloatTensor(obs)
                action = np.ndarray(shape=(1, output_size), buffer=np.array(imitation_net(obs_tensor).data.tolist()))
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


if __name__ == '__main__':
    main()
