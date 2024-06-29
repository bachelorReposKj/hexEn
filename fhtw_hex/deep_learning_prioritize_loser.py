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
from torch.optim.lr_scheduler import StepLR

class ReplayMemory:
    def __init__(self, capacity):
        self.winning_memory = deque(maxlen=capacity // 2)
        self.losing_memory = deque(maxlen=capacity // 2)

    def push(self, state, action, reward, next_state, done, is_winner):
        if is_winner:
            self.winning_memory.append((state, action, reward, next_state, done))
        else:
            self.losing_memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        losing_batch_size = int(batch_size * 0.25)
        winning_batch_size = batch_size - losing_batch_size

        losing_samples = random.sample(self.losing_memory, min(len(self.losing_memory), losing_batch_size))
        winning_samples = random.sample(self.winning_memory, min(len(self.winning_memory), winning_batch_size))

        return losing_samples + winning_samples

    def __len__(self):
        return len(self.winning_memory) + len(self.losing_memory)

def train_dqn(policy, memory, optimizer, criterion, batch_size, gamma, target, device):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

    batch_state = torch.stack(batch_state).to(device)
    batch_action = torch.tensor(batch_action, dtype=torch.long).unsqueeze(1).to(device)
    batch_reward = torch.tensor(batch_reward, dtype=torch.float32).to(device)
    batch_next_state = torch.stack(batch_next_state).to(device)
    batch_done = torch.tensor(batch_done, dtype=torch.float32).to(device)

    current_q_values = policy(batch_state.unsqueeze(1)).gather(1, batch_action).squeeze()
    next_q_values = target(batch_next_state.unsqueeze(1))

    valid_action_mask = (batch_next_state == 0).view(batch_next_state.size(0), -1).float().to(device)
    next_q_values = next_q_values * valid_action_mask + (1 - valid_action_mask) * float(-2)

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

def get_action_no_epsilon(adversary, board, action_set, size=5):
    with torch.no_grad():
        q_values = adversary(board.unsqueeze(0).unsqueeze(0)).view(-1)
        valid_q_values = [q_values[action[0] * size + action[1]] for action in action_set]
        valid_q_values = torch.tensor(valid_q_values)
        return action_set[torch.argmax(valid_q_values).item()]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    size = 5
    num_episodes = 100
    memory = ReplayMemory(50000)
    batch_size = 64
    gamma = 0.99999
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay = 0.9999
    TAU = 0.05

    adversary = DQN(size, size * size)
    filename = "hex_dqn_agent.pth"
    adversary.load_state_dict(torch.load(filename, map_location=device))

    policy = DQN(size, size * size).to(device)
    policy.load_state_dict(torch.load(filename, map_location=device))
    target = DQN(size, size * size).to(device)
    target.load_state_dict(policy.state_dict())

    optimizer = optim.Adam(policy.parameters(), lr=0.01)
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
        episode_done = False

        tmp = hexPosition(size=5)


        while game.winner == 0:
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

            memory.push(state.cpu(), action_space.index(action), reward, next_state.cpu(), done, game.winner == 1)
            state = next_state

            episode_reward += reward

            if done:
                episode_done = True
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

    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"hex_dqn_agent_{current_datetime}.pth"
    torch.save(policy.state_dict(), filename)

    evaluate_agent(target, size, device)

if __name__ == "__main__":
    main()
