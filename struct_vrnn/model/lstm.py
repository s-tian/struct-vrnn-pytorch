import torch
import torch.nn as nn


class MultiGaussianLSTM(nn.Module):
    """Multi layer lstm with Gaussian output."""

    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.embed = nn.Linear(input_size, hidden_size)
        self.mean = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, output_size),
            nn.LeakyReLU(0.2)
        )
        self.logvar = nn.Linear(hidden_size, output_size)
        self.layers_0 = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers
        )

    def forward(self, x, states):
        # assume x to only contain one timestep i.e. (bs, feature_dim)
        x = self.embed(x)
        x = x.view((1,) + x.shape)
        x, new_states = self.layers_0(x, states)
        mean = self.mean(x)[0]
        logvar = self.logvar(x)[0]

        epsilon = torch.normal(mean=0, std=1, size=mean.shape).to(mean.device)
        var = torch.exp(0.5 * logvar)
        z_t = mean + var * epsilon
        return (z_t, mean, logvar), new_states
