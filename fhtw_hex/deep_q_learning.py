import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from hex_engine import hexPosition
import matplotlib.pyplot as plt

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
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


def train_dqn(agent, memory, optimizer, criterion, batch_size, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.stack(batch_state)
    batch_action = torch.tensor(batch_action).unsqueeze(1)
    batch_reward = torch.tensor(batch_reward)
    batch_next_state = torch.stack(batch_next_state)
    batch_done = torch.tensor(batch_done, dtype=torch.float32)

    current_q_values = agent(batch_state).gather(1, batch_action).squeeze()
    max_next_q_values = agent(batch_next_state).max(1)[0]
    expected_q_values = batch_reward + (gamma * max_next_q_values * (1 - batch_done))

    loss = criterion(current_q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def get_state_tensor(board):
    return torch.tensor(board, dtype=torch.float32).view(-1)



def select_action(agent, state, epsilon, action_space,size):
    if random.random() < epsilon:
        return random.choice(action_space)
    else:
        with torch.no_grad():
            q_values = agent(state)
            # Filter out invalid actions by setting their Q-values to a very low value
            valid_q_values = [q_values[action[0]*size+action[1]] for action in action_space]
            valid_q_values = torch.tensor(valid_q_values)
            return action_space[torch.argmax(valid_q_values)]


def main():
    size = 5
    num_episodes = 4000
    memory = ReplayMemory(10000)
    batch_size = 64
    gamma = 0.999
    epsilon_start = 1.0
    epsilon_end = 0.01
    epsilon_decay = 0.9999

    agent = DQN(size*size, size*size)
    optimizer = optim.Adam(agent.parameters())
    criterion = nn.MSELoss()
    episode_rewards = []

    for episode in range(num_episodes):
        game = hexPosition(size)
        state = get_state_tensor(game.board)
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        episode_reward = 0

        while game.winner == 0:
            action_space = game.get_action_space()
            action = select_action(agent, state, epsilon, action_space,size)
            game.moove(action)
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            next_state = get_state_tensor(game.board)
            done = game.winner != 0

            memory.push(state, action_space.index(action), reward, next_state, done)
            state = next_state

            episode_reward += reward

            if done:
                break
        episode_rewards.append(episode_reward)

        train_dqn(agent, memory, optimizer, criterion, batch_size, gamma)

        if episode % 100 == 0:
            print(f'Episode {episode}, Epsilon: {epsilon}')

    # Plot the episode rewards
    window_size = 500
    window = np.ones(int(window_size)) / float(window_size)
    test = np.convolve(episode_rewards, window, 'same')
    plt.plot(test, color = 'r')
    plt.scatter(range(len(episode_rewards)), episode_rewards, s=10)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward Evolution During Training')
    plt.show()

    torch.save(agent.state_dict(), "hex_dqn_agent.pth")

if __name__ == "__main__":
    main()
