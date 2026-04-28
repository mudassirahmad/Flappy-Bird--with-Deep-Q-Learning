import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantage = nn.Linear(hidden_dim, 256)
            self.advantage = nn.Linear(256, output_dim)
        else:

            self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            value = F.relu(self.fc_value(x))
            value = self.value(value)

            advantage = F.relu(self.fc_advantage(x))
            advantage = self.advantage(advantage)

            q_value = value + advantage - torch.mean(advantage, dim=1, keepdim=True)
            return q_value

        else:
            q_value =  self.fc2(x)

        return q_value
    

if __name__ == "__main__":
    input_dim = 12
    output_dim = 2
    net = DQN(input_dim, output_dim)
    state = torch.randn(10, input_dim)
    output = net(state)
    print(output)