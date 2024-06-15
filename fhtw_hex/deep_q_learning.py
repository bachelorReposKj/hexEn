import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from hex_engine import hexPosition
import matplotlib.pyplot as plt
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, board_size, output_dim):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        def conv2d_size_out(size, kernel_size=3, stride=1, padding=1):
            return (size - kernel_size + 2 * padding) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(board_size)))
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        return x


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

from torch.optim.lr_scheduler import StepLR

def train_dqn(policy, memory, optimizer, criterion, batch_size, gamma, target, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.stack(batch_state).to(device)
    batch_action = torch.tensor(batch_action).unsqueeze(1).to(device)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
    batch_next_state = torch.stack(batch_next_state).to(device)
    batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

    current_q_values = policy(batch_state.unsqueeze(1)).gather(1, batch_action).squeeze()
    next_q_values = target(batch_next_state.unsqueeze(1))

    valid_action_mask = (batch_next_state == 0).view(batch_next_state.size(0), -1).float().to(device)
    next_q_values = next_q_values * valid_action_mask + (1 - valid_action_mask) * float(-10)

    max_next_q_values = next_q_values.max(1)[0]
    expected_q_values = batch_reward + (gamma * max_next_q_values * (1 - batch_done))

    loss = criterion(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

def get_state_tensor(board):
    """Convert the 2-dimensional board to a tensor"""
    return torch.tensor(board, dtype=torch.float32)

def select_action(agent, state, epsilon, action_space, size, device):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        with torch.no_grad():
            q_values = agent(state.unsqueeze(0).unsqueeze(0).to(device)).view(-1)
            valid_q_values = [q_values[action[0] * size + action[1]] for action in action_space]
            valid_q_values = torch.tensor(valid_q_values).to(device)
            return action_space[torch.argmax(valid_q_values).item()]

def evaluate_agent(policy, size, device):
    num_simulations = 200
    total_episode_length = 0
    total_reward = 0

    for _ in range(num_simulations):
        game = hexPosition(size)
        state = get_state_tensor(game.board).to(device)
        episode_reward = 0
        episode_length = 0

        while game.winner == 0:
            action_space = game.get_action_space()
            action = select_action(policy, state, 0, action_space, size, device)
            game.moove(action)
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            if reward == 0:
                game._random_moove()  # for now, the opponent has the random strategy
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            state = get_state_tensor(game.board).to(device)

            episode_reward += reward
            episode_length += 1

            if game.winner != 0:
                break

        total_episode_length += episode_length
        total_reward += episode_reward


    average_episode_length = total_episode_length / num_simulations
    average_reward = total_reward / num_simulations

    print(f'Average Episode Length: {average_episode_length}')
    print(f'Average Reward: {average_reward}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = 5
    num_episodes = 20000
    memory = ReplayMemory(50000)
    batch_size = 64
    gamma = 0.99999
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.9995
    TAU = 0.05

    policy = DQN(size, size * size).to(device)
    target = DQN(size, size * size).to(device)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters())
    scheduler = StepLR(optimizer, step_size=1000, gamma=0.99)
    criterion = nn.MSELoss()
    episode_rewards = []
    losses = []
    update_frequency = 100

    for episode in range(num_episodes):
        game = hexPosition(size)
        state = get_state_tensor(game.board).to(device)
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        episode_reward = 0


        while game.winner == 0:
            action_space = game.get_action_space()
            action = select_action(policy, state, epsilon, action_space, size, device)
            game.moove(action)
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            if reward == 0:
                game._random_moove()  # for now, the opponent has the random strategy
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            next_state = get_state_tensor(game.board).to(device)
            done = game.winner != 0

            memory.push(state.cpu(), action_space.index(action), reward, next_state.cpu(), done)
            state = next_state

            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

        loss = train_dqn(policy, memory, optimizer, criterion, batch_size, gamma, target, device)
        losses.append(loss)
        scheduler.step()

        if num_episodes % update_frequency == 0:
            target.load_state_dict(policy.state_dict())

        if episode % 100 == 0:
            print(f'Episode {episode}, Epsilon: {epsilon}, Loss: {loss}')

    window_size = num_episodes / 100
    window = np.ones(int(window_size)) / float(window_size)
    test = np.convolve(episode_rewards, window, 'valid')
    plt.plot(test, color='r')
    plt.scatter(range(len(episode_rewards)), episode_rewards, s=10)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Evolution During Training')
    plt.show()


    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Evolution During Training')
    plt.show()

    torch.save(policy.state_dict(), "hex_dqn_agent.pth")

    # Run simulation to evaluate the trained agent
    evaluate_agent(target, size, device)



if __name__ == "__main__":
    main()



