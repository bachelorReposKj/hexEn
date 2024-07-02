
import torch.nn as nn
import random
from fhtw_hex import hex_engine
import torch.nn.functional as F
import torch
import os

def get_state_tensor(board):
    """Convert the 2-dimensional board to a tensor"""
    return torch.tensor(board, dtype=torch.float32)

class ChannelSplitter(nn.Module):
    def __init__(self):
        super(ChannelSplitter, self).__init__()

    def forward(self, x):
        white_stones = (x == 1).float()
        black_stones = (x == -1).float()
        return torch.cat([white_stones, black_stones], dim=1)
class DQN(nn.Module):
    def __init__(self, board_size, output_dim):
        super(DQN, self).__init__()
        self.splitter = ChannelSplitter()  # Add the custom layer
        self.conv1 = nn.Conv2d(2, 512, kernel_size=3, stride=1, padding=1)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        linear_input_size = convw * convh * 512

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = self.splitter(x)  # Use the custom layer
        x = F.leaky_relu(self.conv1(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def select_action(agent, state, epsilon, action_space, size, device):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        with torch.no_grad():
            q_values = agent(state.unsqueeze(0).unsqueeze(0).to(device)).view(-1)
            valid_q_values = [q_values[action[0] * size + action[1]] for action in action_space]
            valid_q_values = torch.tensor(valid_q_values).to(device)
            return action_space[torch.argmax(valid_q_values).item()]





class agent:
    def __init__(self, model_path="hex_dqn_agent_2024-06-25_11-43-39.pth", size=7):
        script_dir = os.path.dirname(__file__)  # Get the directory of the current script
        model_path = os.path.join(script_dir, model_path)  # Construct the full path to the model file

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"The model file was not found: {model_path}")

        self.model = DQN(size, size * size)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        self.model.eval()
        self.size = size

    def __call__(self, board, action_set):
    #Assumes the agent is playing for white
        state = get_state_tensor(board)

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0).unsqueeze(0)).view(-1)
            valid_q_values = [q_values[action[0] * self.size + action[1]] for action in action_set]
            valid_q_values = torch.tensor(valid_q_values)
            best_action = action_set[torch.argmax(valid_q_values).item()]

        return best_action


game = hex_engine.hexPosition(7)
game.machine_vs_machine(None,agent())