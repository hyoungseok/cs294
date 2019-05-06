import numpy as np
import pickle
import tensorflow as tf
import load_policy
import tf_util
from train_imitation import train
from run_imitation import run
from shutil import copyfile

# 롤아웃으로 이미테이션 에이전트 학습 (잘 안됨)
# 이미테이션 에이전트의 옵져베이션을 저장
# 익스퍼트에게 에이전트의 옵져베이션을 주고 액션을 받아서 롤아웃으로 저장
# 이 롤아웃으로 이미테이션 에이전트 학습

task_list = [
    # "Hopper-v2",
    # "Ant-v2",
    # "HalfCheetah-v2",
    # "Humanoid-v2",
    "Reacher-v2",
    # "Walker2d-v2",
]


def init_dagger(task, train_path):
    source = f"{train_path}/{task}.pkl"
    target = f"dagger_rollout/{task}.pkl"
    copyfile(source, target)


def ask(task, obs_path):
    print(f"task:{task}")
    policy_fn = load_policy.load_policy(f"experts/{task}.pkl")

    with open(f"{obs_path}/{task}.pkl", "rb") as obs_file:
        read_obs = pickle.loads(obs_file.read())
        observations = read_obs["observations"]

    actions = []
    with tf.Session():
        tf_util.initialize()
        for obs in observations:
            action = policy_fn(obs[None, :])
            actions.append(action)

    with open(f"dagger_rollout/{task}.pkl", "rb") as dagger_read_file:
        read_data = pickle.loads(dagger_read_file.read())
        dagger_obs = np.concatenate((read_data["observations"], observations), axis=0)
        dagger_action = np.concatenate((read_data["actions"], np.array(actions)), axis=0)

    dagger_data = {"observations": dagger_obs, "actions": dagger_action}

    with open(f"dagger_rollout/{task}.pkl", "wb") as dagger_write_file:
        pickle.dump(dagger_data, dagger_write_file, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    max_dagger = 5
    for task_name in task_list:
        init_dagger(task_name, "experts_rollout")
        for i in range(max_dagger):
            print(f"dagger count : {i + 1}")
            train(task_name, "dagger_rollout", "dagger")
            run(task_name, "dagger")
            ask(task_name, "imitations_obs")
