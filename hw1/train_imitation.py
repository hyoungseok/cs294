import torch
from torch import nn
import torch.optim as optim
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from imitaion_model import Net

task_list = [
    "Hopper-v2",
    "Ant-v2",
    "HalfCheetah-v2",
    "Humanoid-v2",
    "Reacher-v2",
    "Walker2d-v2",
]

batch_size = 1024
epoch = 200
test_rate = 0.2

for task in task_list:
    print(f"task : {task}")
    with open(f"experts_rollout/{task}.pkl", "rb") as f:
        experts_data = pickle.loads(f.read())
        x_data = experts_data["observations"]
        y_data = []
        for action_data in experts_data["actions"]:
            y_data.append(action_data[0])

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_rate, random_state=42)

    input_size = len(x_train[0])
    hidden_size = input_size
    output_size = len(y_train[0])

    print(f"train data X : {len(x_train)}")
    print(f"train data Y : {len(y_train)}")
    print(f"test data X : {len(x_test)}")
    print(f"test data Y : {len(y_test)}")

    print(f"input size : {input_size}")
    print(f"output size : {output_size}")

    data_size = len(x_train)
    iterations = int(data_size / batch_size + 0.5) * epoch
    print(f"total iterations : {iterations}")

    net = Net(input_size, hidden_size, output_size)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    x_test_tensor = torch.FloatTensor(x_test)
    y_test_tensor = torch.FloatTensor(y_test)

    loss_history = []
    if len(x_data) > 100 * (1 + test_rate):
        checkpoint = 100
    else:
        checkpoint = len(x_data) * test_rate

    for i in range(iterations):

        net.train()
        replacement = batch_size > len(x_train)
        batch_idx = np.random.choice(len(x_train), batch_size, replacement).astype(int)
        x_batch = []
        y_batch = []
        for idx in batch_idx:
            x_batch.append(x_train[idx])
            y_batch.append(y_train[idx])

        x_batch_tensor = torch.FloatTensor(x_batch)
        y_batch_tensor = torch.FloatTensor(y_batch)

        optimizer.zero_grad()
        train_pred = net(x_batch_tensor)
        train_loss = criterion(train_pred, y_batch_tensor)
        train_loss.backward()
        optimizer.step()

        if (i + 1) % checkpoint == 0:
            net.eval()
            test_pred = net(x_test_tensor)
            test_loss = criterion(test_pred, y_test_tensor)
            loss_history.append([i + 1, test_loss.item()])
            print(f"{i + 1:6d} iter || train loss {train_loss.item():.5f} || test loss {test_loss.item():.5f}")

    best_iterations = min(loss_history, key=lambda x: x[1])[0]
    print(f"best iterations : {best_iterations}")
    for j in range(best_iterations):

        net.train()
        replacement = batch_size > len(x_data)
        batch_idx = np.random.choice(len(x_data), batch_size, replacement).astype(int)
        x_batch = []
        y_batch = []
        for idx in batch_idx:
            x_batch.append(x_data[idx])
            y_batch.append(y_data[idx])

        x_batch_tensor = torch.FloatTensor(x_batch)
        y_batch_tensor = torch.FloatTensor(y_batch)

        optimizer.zero_grad()
        train_pred = net(x_batch_tensor)
        train_loss = criterion(train_pred, y_batch_tensor)
        train_loss.backward()
        optimizer.step()

        if (j + 1) % checkpoint == 0:
            print(f"{j + 1:6d} iter || train loss {train_loss.item():.5f}")
        if j + 1 == best_iterations:
            torch.save(net.state_dict(), f"imitations/{task}.pth")
            with open(f"imitations/{task}.conf", "w") as model_conf:
                model_conf.write(f"input_size={input_size}\nhidden_size={hidden_size}\noutput_size={output_size}")
            print(f"{task} model saved")
