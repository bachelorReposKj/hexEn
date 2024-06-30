# Hex Game AI with Deep Q-Learning

This repository contains the implementation of a Deep Q-Learning (DQN) approach for developing a competitive AI agent for the game Hex.

## Overview

Hex is a two-player abstract strategy board game played on a hexagonal grid. The goal is to connect two opposite sides of the board with a continuous path of your colored pieces before your opponent does the same.

### Key Features

- **Deep Q-Learning**: Implemented using Python and PyTorch to train an AI agent.
- **Prioritized Experience Replay**: Enhances learning efficiency by prioritizing important experiences.
- **Dual-Channel State Representation**: Enables more accurate state evaluations for better decision-making.
- **Handling Invalid Actions**: Strategy to prevent the agent from considering invalid moves, improving decision-making accuracy.
- **Information about the files**: There have been different files where the training took place, but in the end we focused on using the following file -> deep_learning_prioritized_replay.py
  
## Installation

To run the code, follow these steps:

1. Clone the repository:

   ```
   bash
   git clone https://github.com/gucccikev/hexEn.git
   cd hex-game-ai```
2. Install dependencies:

    ```pip install -r requirements.txt```


3. Run the training script:

    ```python deep_learning_prioritized_replay.py```
