import torch
import random
from hex_engine import hexPosition
from dqn import DQN
from deep_q_learning import select_action

class AgentWrapper:
    def __init__(self, model, size):
        self.model = model
        self.size = size

    def __call__(self, board, action_set):
        state = get_state_tensor(board)

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0).unsqueeze(0)).view(-1)
            valid_q_values = [q_values[action[0] * self.size + action[1]] for action in action_set]
            valid_q_values = torch.tensor(valid_q_values)
            return action_set[torch.argmax(valid_q_values).item()]

    def __call__(self, board, action_set):
        # Determine if it is black's turn by summing the board
        board_sum = sum(sum(row) for row in board)
        is_blacks_turn = board_sum != 0

        if is_blacks_turn:
            # Create a temporary hexPosition object to use the recoding functions
            temp = hexPosition(size=self.size)
            temp.board = board
            recoded_board = temp.recode_black_as_white()
            recoded_action_set = [temp.recode_coordinates(action) for action in action_set]
        else:
            recoded_board = board
            recoded_action_set = action_set

        state = get_state_tensor(recoded_board)

        with torch.no_grad():
            q_values = self.model(state.unsqueeze(0).unsqueeze(0)).view(-1)
            valid_q_values = [q_values[action[0] * self.size + action[1]] for action in recoded_action_set]
            valid_q_values = torch.tensor(valid_q_values)
            best_action = recoded_action_set[torch.argmax(valid_q_values).item()]

        if is_blacks_turn:
            # Recode the best action back to black's perspective
            best_action = temp.recode_coordinates(best_action)

        return best_action


def get_state_tensor(board):
    """Convert the 2-dimensional board to a tensor"""
    return torch.tensor(board, dtype=torch.float32)

def random_agent(board, action_set):
    return random.choice(action_set)

def machine_vs_machine_wrapper(size, model_white, model_black, play_against_random=False):
    # Create agent wrappers for both models
    agent_white = AgentWrapper(model_white, size)
    agent_black = AgentWrapper(model_black, size)

    # Create a Hex game instance
    game = hexPosition(size)

    if play_against_random:
        # Run the machine_vs_machine function with one agent and random agent
        game.machine_vs_machine(machine1=agent_white, machine2=random_agent)
    else:
        # Run the machine_vs_machine function with both agents
        game.machine_vs_machine(machine1=agent_white, machine2=agent_black)


def human_vs_machine_wrapper(size, human_player, model):
    # Create agent wrappers for both models

    # Create a Hex game instance

    game = hexPosition(size)

    agent_white = AgentWrapper(model, size)

    game.human_vs_machine( human_player=human_player, machine = agent_white)

def evaluate_agent(policy, size):
    num_simulations = 20000
    total_episode_length = 0
    total_reward = 0

    for _ in range(num_simulations):
        game = hexPosition(size)
        state = get_state_tensor(game.board)
        episode_reward = 0
        episode_length = 0

        while game.winner == 0:
            action_space = game.get_action_space()
            action = select_action(policy, state, 0, action_space, size)
            game.moove(action)
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            if reward == 0:
                game._random_moove()  # for now, the opponent has the random strategy
            reward = 1 if game.winner == 1 else -1 if game.winner == -1 else 0
            state = get_state_tensor(game.board)

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


if __name__ == "__main__":
    size = 7  # Board size
    model_white_path = "hex_dqn_agent_2024-06-25_10-34-03.pth"
    model_black_path = "hex_dqn_agent_2024-06-25_10-34-03.pth"

    # Load the models
    model_white = DQN(size, size * size)
    model_white.load_state_dict(torch.load(model_white_path, map_location=torch.device('cpu')))
    model_white.eval()

    model_black = DQN(size, size * size)
    model_black.load_state_dict(torch.load(model_black_path, map_location=torch.device('cpu')))
    model_black.eval()

    # Choose whether to play against random agent
    play_against_random = False # Set t# o False to play agents against each other


    play_against_human = True
    # Run the competition

    evaluate = False

    if evaluate == True:
        evaluate_agent(model_white,5)
    elif play_against_human:
        human_vs_machine_wrapper(size, human_player=-1, model = model_white)

    else:
        machine_vs_machine_wrapper(size, model_white, model_black, play_against_random)
