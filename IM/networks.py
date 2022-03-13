import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, SAGEConv, TopKPooling
from torch.utils.data import Dataset
import copy


class KPooling(nn.Module):

    def __init__(self, ratio, input_size, hidden_size, output_size):

        super(KPooling, self).__init__()

        self.gcn1 = SAGEConv(input_size, hidden_size)
        self.first_pool_layer = TopKPooling(hidden_size, ratio)
        self.gcn2 = SAGEConv(hidden_size, hidden_size)
        self.second_pool_layer = TopKPooling(hidden_size, ratio)
        self.gcn3 = SAGEConv(hidden_size, hidden_size)
        self.third_pool_layer = TopKPooling(hidden_size, ratio)
        self.output_layer = nn.Linear(hidden_size * 2, output_size)

    def forward(self, data, return_node_embedding=False, return_both=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.gcn1(x, edge_index)
        if return_node_embedding and not return_both:
            return x
        elif return_both:
            x_ = copy.deepcopy(x.detach())
        x, edge_index, _, batch, _, _ = self.first_pool_layer(x, edge_index, batch=batch)
        summary_1_mean = global_mean_pool(x, batch)
        summary_1_max = global_max_pool(x, batch)
        summary_1 = torch.cat((summary_1_mean, summary_1_max), dim=1)

        x = torch.relu(self.gcn2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.second_pool_layer(x, edge_index, batch=batch)
        summary_2_mean = global_mean_pool(x, batch)
        summary_2_max = global_max_pool(x, batch)
        summary_2 = torch.cat((summary_2_mean, summary_2_max), dim=1)

        x = self.gcn3(x, edge_index)
        x, edge_index, _, batch, _, _ = self.third_pool_layer(x, edge_index, batch=batch)
        summary_3_mean = global_mean_pool(x, batch)
        summary_3_max = global_max_pool(x, batch)
        summary_3 = torch.cat((summary_3_mean, summary_3_max), dim=1)

        summary = summary_1 + summary_2 + summary_3

        summary = self.output_layer(summary)

        if not return_both:
            return summary
        elif return_both:
            return summary, x_


class CustomDataset(Dataset):
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()

        self.input = nn.Linear(input_size, 2)
        self.output = nn.Linear(2, input_size)

    def forward(self, x, embedding=False):
        x = self.input(x)
        if embedding:
            return x
        x = self.output(x)
        return x


class QNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(QNet, self).__init__()

        self.state_input = nn.Linear(state_dim, hidden_size)
        self.action1_input = nn.Linear(action_dim, hidden_size)
        self.action2_input = nn.Linear(action_dim, hidden_size)

        self.h1 = nn.Linear(3 * hidden_size, hidden_size)
        self.h2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, 1)

        self.state_dim = state_dim
        self.action_dim = action_dim

    def forward(self, x):
        state = self.state_dim
        action1 = self.state_dim + self.action_dim
        if len(x.shape) == 2:
            state, action1, action2 = x[:, :state], x[:, state:action1], x[:, action1:]
            state = torch.relu(self.state_input(state))
            action1 = torch.relu(self.action1_input(action1))
            action2 = torch.relu(self.action2_input(action2))
            x = torch.cat((state, action1, action2), dim=1)
        elif len(x.shape) == 3:
            state, action1, action2 = x[:, :, :state], x[:, :, state:action1], x[:, :, action1:]
            state = torch.relu(self.state_input(state))
            action1 = torch.relu(self.action1_input(action1))
            action2 = torch.relu(self.action2_input(action2))
            x = torch.cat((state, action1, action2), dim=2)

        x = torch.relu(self.h1(x))
        x = torch.relu(self.h2(x))
        x = self.output_layer(x)
        return x
