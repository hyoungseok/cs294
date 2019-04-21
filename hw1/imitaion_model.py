import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Net, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.Tanh(),
            nn.Linear(input_dim * 2, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, int((input_dim + output_dim) / 2 + 0.5)),
            nn.Tanh(),
            nn.Linear(int((input_dim + output_dim) / 2 + 0.5), output_dim)
        )

    def forward(self, x):
        return self.linear(x)
