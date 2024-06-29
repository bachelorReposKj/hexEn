import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from hex_engine import hexPosition
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dqn import DQN, hexPosition
from datetime import datetime
import random

class PrioritizedReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha

    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities, default=1.0)
        self.memory.append((state, action, reward, next_state, done))
        self.priorities.append(max_priority)

    def sample(self, batch_size, beta=0.4):
        if len(self.memory) == self.capacity:
            priorities = np.array(self.priorities)
        else:
            priorities = np.array(self.priorities)[:len(self.memory)]

        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[idx] for idx in indices]

        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()

        batch = list(zip(*samples))
        states = torch.stack(batch[0])
        actions = torch.tensor(batch[1]).unsqueeze(1)
        rewards = torch.tensor(batch[2], dtype=torch.float32)
        next_states = torch.stack(batch[3])
        dones = torch.tensor(batch[4], dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.memory)

from torch.optim.lr_scheduler import StepLR

def train_dqn(policy, memory, optimizer, criterion, batch_size, gamma, target, device, beta=0.4):
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states, dones, indices, weights = memory.sample(batch_size, beta)
    states = states.to(device)
    actions = actions.to(device)
    rewards = rewards.to(device)
    next_states = next_states.to(device)
    dones = dones.to(device)
    weights = weights.to(device)

    current_q_values = policy(states.unsqueeze(1)).gather(1, actions).squeeze()
    next_q_values = target(next_states.unsqueeze(1))

    valid_action_mask = (next_states == 0).view(next_states.size(0), -1).float().to(device)
    next_q_values = next_q_values * valid_action_mask + (1 - valid_action_mask) * float(-2)

    max_next_q_values = next_q_values.max(1)[0]
    expected_q_values = rewards + (gamma * max_next_q_values * (1 - dones))

    loss = (weights * criterion(current_q_values, expected_q_values)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    errors = torch.abs(current_q_values - expected_q_values).cpu().detach().numpy()
    memory.update_priorities(indices, errors + 1e-6)

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
    num_simulations = 2000
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

def get_action_no_epsilon (adversary,board, action_set,size = 5):
    with torch.no_grad():
        q_values = adversary(board.unsqueeze(0).unsqueeze(0)).view(-1)
        valid_q_values = [q_values[action[0] * size + action[1]] for action in action_set]
        valid_q_values = torch.tensor(valid_q_values)
        return action_set[torch.argmax(valid_q_values).item()]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = 7
    num_episodes = 20000
    memory = PrioritizedReplayMemory(50000)
    batch_size = 64
    gamma = 0.99999
    epsilon_start = 1.0
    epsilon_end = 0.15
    epsilon_decay = 0.99999
    TAU = 0.05

    adversary = DQN(size, size * size)
    filename = "hex_dqn_agent_2024-06-26_16-58-21.pth"
    adversary.load_state_dict(torch.load(filename))

    policy = DQN(size, size * size).to(device)
    policy.load_state_dict(torch.load("hex_dqn_agent_2024-06-26_16-58-21.pth"))
    target = DQN(size, size * size).to(device)
    target.load_state_dict(policy.state_dict())

    lr = 0.001
    gamma_lr = 0.999
    step_size = 1000
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma_lr)
    criterion = nn.MSELoss()
    episode_rewards = []
    losses = []
    update_frequency = 100

    tmp = hexPosition(size)

    for episode in range(num_episodes):
        game = hexPosition(size)
        state = get_state_tensor(game.board).to(device)
        epsilon = max(epsilon_end, epsilon_start * (epsilon_decay ** episode))
        episode_reward = 0

        while game.winner == 0:
            if random.random() < 0.5: #flip the board by 180 degrees half of the time
                game.flip_board_180()
                state = get_state_tensor(game.board).to(device)
            action_space = game.get_action_space()
            action = select_action(policy, state, epsilon, action_space, size, device)
            game.moove(action)
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            if reward == 0:
                action = get_action_no_epsilon(adversary, get_state_tensor(tmp.recode_black_as_white(game.board)), game.get_action_space(recode_black_as_white=True))
                action = tmp.recode_coordinates(action)
                game.moove(action)
                #game._random_moove()
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

        if episode % update_frequency == 0:
            target.load_state_dict(policy.state_dict())

        if episode % 100 == 0:
            print(f'Episode {episode}, Epsilon: {epsilon}, Loss: {loss}')

    window_size = num_episodes / 100
    window = np.ones(int(window_size)) / float(window_size)
    test = np.convolve(episode_rewards, window, 'valid')
    plt.plot(test, color='r')
    plt.scatter(range(len(episode_rewards)), episode_rewards, marker='o', color='b')
    plt.title(f'Convolution with window size = {window_size}')
    plt.show()

    plt.plot(losses)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('Loss Evolution During Training')
    plt.show()


    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"hex_dqn_agent_{current_datetime}.pth"
    torch.save(policy.state_dict(), filename)
    evaluate_agent(policy, size, device)

if __name__ == "__main__":
    main()
