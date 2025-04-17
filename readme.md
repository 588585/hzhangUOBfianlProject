# Vision-Based Cooperative DQN Agent in Game

This repository contains the implementation of vision-based Deep Q-Network (DQN) and Double DQN (DDQN) agents that can cooperate with simulated human agents in the "Pig Chase" game environment on the Malmo platform.

## Project Overview

This project explores a vision-based cooperative agent model for human-machine interaction in collaborative game environments. Unlike existing solutions that rely on symbolic representations or handcrafted features, our approach learns directly from raw first-person visual input.

## Installation

### Prerequisites
- win 11
- [Python](https://www.python.org/) Python 3.6.5 
- [Project Malmo](https://github.com/Microsoft/malmo) - Malmo-0.36.0-Python3.6
- [Malmo Challenge](https://github.com/Microsoft/malmo-challenge) with (pip install -e '.[all]')

### pip Install

py -3.6 -m pip install -r requirements.txt

### Install to Malmo Challenge

Replace malmopy\environment\malmo\malmo.py with malmo.py

Overwrite all other files to malmo-challenge\ai_challenge\pig_chase

### Installation complete

## Project Structure

```
.
├── common.py                  # Common utility functions and classes
├── environment.py             # Environment configuration and setup
├── malmo.py                   # Interface with Malmo platform
├── MyDQNAgent.py              # Early experimental code is now only used to read models.
├── MyDDQNAgent_in.py          # Experiment configuration and running script
├── MyDDQNAgent.py             # Main DDQN agent implementation (PyTorch)
├── MyDQNExplorer.py           # Explorer implementation for action selection
├── Myevaluation.py            # Evaluation and testing methods
├── MyExtra.py                 # Additional utilities and helper functions
├── pig_chase-dqn_3.model      # Pre-trained dqn model file 300,000 steps
├── model_15000.pt             # Pre-trained ddqn model file 15000 steps
├── pig_chase.xml              # Malmo mission XML configuration
├── readme.md                  # Project documentation
├── requirements.txt           # Dependencies for installation
└── visualization.ipynb        # Jupyter notebook for visualizing results

```

## Usage

### launch Client

``` 
cd Malmo-0.36.0-Windows-64bit_withBoost_Python3.6\Minecraft

.\launchClient.bat -port 10000
.\launchClient.bat -port 10001
```


### Training

```python
# Example code to train a DDQN agent
py -3.6 MyDDQNAgent_in.py 
```

### Evaluation

```python
# Example code to evaluate a trained model
py -3.6 MyDDQNAgent_in.py 
```


### tensorboard

```
tensorboard --logdir=results --port=6006
```




### visualization

folow visualization.ipynb 


