

"""
Pig Chase DDQN Experiment Runner
--------------------------------
This script sets up and runs a Double Deep Q-Network (DDQN) experiment
in the Minecraft Pig Chase environment from Project Malmo.
"""

import os
import sys
import time
from argparse import ArgumentParser
from datetime import datetime
import keyboard
import six
from os import path
from threading import Thread, active_count
from time import sleep

# Configure paths
sys.path.insert(0, os.getcwd())
sys.path.insert(1, os.path.join(os.path.pardir, os.getcwd()))

# Import Malmo environment components
from common import parse_clients_args, visualize_training, ENV_AGENT_NAMES
from agent import PigChaseChallengeAgent
from malmopy.model import QModel

from MyDQNExplorer import UCBExplorer, QuadraticEpsilonGreedyExplorer
from environment import PigChaseEnvironment, PigChaseSymbolicStateBuilder
from malmopy.environment.malmo import MalmoALEStateBuilder
from malmopy.agent import (
    TemporalMemory, 
    RandomAgent, 
    LinearEpsilonGreedyExplorer
)

# Import visualization tools
try:
    from malmopy.visualization.tensorboard import TensorboardVisualizer
    from malmopy.visualization.tensorboard.cntk import CntkConverter
    USING_TENSORBOARD = True
except ImportError:
    print('Cannot import tensorboard, using ConsoleVisualizer.')
    from malmopy.visualization import ConsoleVisualizer
    USING_TENSORBOARD = False

# Constants
EXPERIMENT_NAME = "pig_chase_ddqn"
RESULTS_FOLDER = "results/pig_chase/ddqn"
EPOCH_SIZE = 100000
DEFAULT_CLIENTS = ['127.0.0.1:10000', '127.0.0.1:10001']
MODEL_SAVE_INTERVAL = 5000

# Global variables
slow_speed = False

_config = {
    'batch_size': 32,                  # Batch size for training
    'gamma': 0.99,                     # Discount factor
    'train_after': 5000,               # Start training after this many steps
    'target_update_freq': 100,         # Update target network every N steps
    'train_frequency': 8,              # Train every N steps
    'tau': 0.9                        # Soft update parameter (1.0 = hard update)
}

_explorer = LinearEpsilonGreedyExplorer(
    1.0,        # Start with 100% random actions
    0.1,          # End with 10% random actions
    50000        # Linear decay over 5000 steps
)


def create_ddqn_agent(name, available_actions, memory, backend, device, 
                     model_file=None, visualizer=None, config=_config , explorer=_explorer):
    """
    Create a DDQN agent with appropriate model and explorer
    
    Args:
        name: Agent name
        available_actions: List of available actions
        memory: Replay memory instance
        backend: Neural network backend ('cntk', 'pytorch', etc.)
        device: Device to run the model on
        model_file: Path to pre-trained model (if any)
        visualizer: Visualization tool
        
    Returns:
        agent: Configured DDQN agent
        is_evaluating: Whether agent is in evaluation mode
    """
    # Initialize model based on backend
    if backend == 'cntk':
        print('Using CNTK backend')
        from malmopy.model.cntk import QNeuralNetwork
        from MyDDQNAgent_cntk  import MyDDQN_Q
        model = QNeuralNetwork((memory.history_length, 84, 84), available_actions, device)

    elif backend == 'pytorch':
        from MyDDQNAgent import MyDDQN_Q, PyTorchQModel
        print('Using PyTorch backend')
        model = PyTorchQModel((memory.history_length, 84, 84), available_actions)
        print('PyTorch model created')
    else:
        print('backend:', backend,'backend error')
        raise ValueError(f"Unsupported backend: {backend}")
    
    print(f'Model initialized with history length: {memory.history_length}')
    
    # Create explorer for action selection

    
    # Determine if we're loading a model for evaluation
    is_evaluating = False
    if model_file:
        try:
            model.load(model_file)
            print(f'Model "{model_file}" loaded successfully')
            is_evaluating = True
            print('Running in evaluation mode (explorer disabled)')
        except Exception as e:
            print(f'Failed to load model "{model_file}": {e}')
            print('Running in training mode with new model')
    else:
        print('Running in training mode with new model')
    
    # Create and return the agent
    agent = MyDDQN_Q(
        name=name,
        actions=available_actions,
        model=model, 
        memory=memory, 
        gamma=config['gamma'],
        batch_size=config['batch_size'],
        target_network_train_frequency=config['target_update_freq'],
        explorer=explorer,
        visualizer=visualizer,
        backend=backend,
        My_train_after=config['train_after'],
        is_evaluating=is_evaluating,
        train_frequency=config['train_frequency'],
        tau=config['tau']
    )
    
    return agent, is_evaluating

def run_challenge_agent(name, clients, visualizer, human_speed=False):
    """
    Run the challenge agent (random/focused)
    
    Args:
        name: Agent name
        clients: Minecraft client endpoints
        visualizer: Visualization tool
        human_speed: Whether to run at human-observable speed
    """
    print(f'Starting challenge agent: {name}')
    
    # Initialize environment with symbolic state builder
    builder = PigChaseSymbolicStateBuilder()

    env = PigChaseEnvironment(
        clients,
        builder,
        role=0,  # Challenge agent role
        human_speed=human_speed,
        randomize_positions=True
    )

    
    # Create challenge agent
    agent = PigChaseChallengeAgent(name)
    
    # Determine agent type
    if isinstance(agent.current_agent, RandomAgent):
        agent_type = PigChaseEnvironment.AGENT_TYPE_1
    else:
        agent_type = PigChaseEnvironment.AGENT_TYPE_2
    
    # Initial reset
    obs = env.reset(agent_type)
    reward = 0
    agent_done = False
    
    # Main agent loop
    while True:
        # Slow down execution if needed
        if human_speed or slow_speed:
            sleep(0.1)
            
        # Episode completion
        if env.done:
            # Re-determine agent type
            if isinstance(agent.current_agent, RandomAgent):
                agent_type = PigChaseEnvironment.AGENT_TYPE_1
            else:
                agent_type = PigChaseEnvironment.AGENT_TYPE_2
                
            # Reset environment
            obs = env.reset(agent_type)
            while obs is None:
                print('Warning: received obs=None, resetting again.')
                obs = env.reset(agent_type)
        
        # Select action and step environment
        action = agent.act(obs, reward, agent_done, is_training=True)
        obs, reward, agent_done = env.do(action)

def run_ddqn_agent(name, clients, backend, device, max_epochs,
                 logdir, visualizer, human_speed=False, model_file=None):
    """
    Run the DDQN learning agent
    
    Args:
        name: Agent name
        clients: Minecraft client endpoints
        backend: Neural network backend
        device: Device to run the model on
        max_epochs: Maximum number of epochs to run
        logdir: Log directory
        visualizer: Visualization tool
        human_speed: Whether to run at human-observable speed
        model_file: Path to pre-trained model (if any)
    """
    print(f'Starting DDQN agent: {name} with {backend} backend')
    
    # Initialize environment with ALE state builder

    env = PigChaseEnvironment(
        clients,
        state_builder=MalmoALEStateBuilder(),
        role=1,  # DDQN agent role
        human_speed=human_speed,
        randomize_positions=True
    )

    # Create replay memory
    memory = TemporalMemory(
        max_size=100000,    # Store 100k experience tuples
        sample_shape=(84, 84)  # 84x84 grayscale images
    )
    


    # Create DDQN agent
    agent, is_evaluating = create_ddqn_agent(
        name=name,
        available_actions=env.available_actions,
        memory=memory,
        backend=backend,
        device=device,
        model_file=model_file,
        visualizer=visualizer
    )
    
    # Initialize training
    print('Getting first observation...')
    obs = env.reset()
    reward = 0
    agent_done = False
    viz_rewards = []
    
    # Calculate total training steps
    max_training_steps = EPOCH_SIZE * max_epochs
    
    print(f'Starting DDQN training for {max_training_steps} steps')
    
    # Main training loop
    for step in six.moves.range(1, max_training_steps + 1):
        # Slow down execution if needed
        if human_speed or slow_speed:
            sleep(0.1)
            
        # Episode completion
        if env.done:
            # Log performance metrics
            visualize_training(visualizer, step, viz_rewards)
            agent.inject_summaries(step)
            viz_rewards = []
            
            # Reset environment
            obs = env.reset()
            while obs is None:
                print('Warning: received obs=None, resetting again.')
                obs = env.reset()
        
        # Select action and step environment
        action = agent.act(obs, reward, agent_done, is_training=True)
        next_obs, reward, agent_done = env.do(action)
        
        # Track rewards for visualization
        viz_rewards.append(reward)
        obs = next_obs
        
        # Periodic reporting
        if step % 200 == 0:
            progress = step / max_training_steps * 100
            print(f'Step {step}/{max_training_steps} ({progress:.1f}%)')
            
        # Periodic model saving
        if step % MODEL_SAVE_INTERVAL == 0:
            model_path = os.path.join(logdir, f'model_{step}.pt')
            agent._model.save(model_path)
            print(f'Model saved to {model_path}')

def create_visualizer(logdir):
    """
    Create an appropriate visualizer
    
    Args:
        logdir: Log directory
        
    Returns:
        visualizer: Configured visualizer
    """
    if USING_TENSORBOARD:
        visualizer = TensorboardVisualizer()
        visualizer.initialize(logdir, None)
    else:
        visualizer = ConsoleVisualizer()
    
    return visualizer

def run_experiment(agents_def):
    """
    Run the experiment with two agents
    
    Args:
        agents_def: List of agent definitions
    """
    assert len(agents_def) == 2, f'Need exactly 2 agents (got: {len(agents_def)})'
    
    # Initialize experiment
    processes = []
    global slow_speed
    slow_speed = False
    
    # Optional: Keyboard shortcut to toggle simulation speed
    # def toggle_speed():
    #     global slow_speed
    #     slow_speed = not slow_speed
    #     print(f"Simulation speed toggled to: {'slow' if slow_speed else 'fast'}")
    # keyboard.add_hotkey('ctrl+b', toggle_speed)
    
    # Start agent threads
    for agent_def in agents_def:
        if agent_def['role'] == 0:
            thread_target = run_challenge_agent
            thread_kwargs = {
                'name': agent_def['name'],
                'clients': agent_def['clients'],
                'visualizer': agent_def['visualizer'],
                'human_speed': agent_def['human_speed']
            }
        else:
            thread_target = run_ddqn_agent
            thread_kwargs = {
                'name': agent_def['name'],
                'clients': agent_def['clients'],
                'backend': agent_def['backend'],
                'device': agent_def['device'],
                'max_epochs': agent_def['max_epochs'],
                'logdir': agent_def['logdir'],
                'visualizer': agent_def['visualizer'],
                'human_speed': agent_def['human_speed'],
                'model_file': agent_def['model_file']
            }
        
        # Create and start thread
        p = Thread(target=thread_target, kwargs=thread_kwargs)
        p.daemon = True
        p.start()
        
        # Give the server time to start if it's the challenge agent
        if agent_def['role'] == 0:
            sleep(2)
            
        processes.append(p)
    
    try:
        # Wait until only the main thread is left
        while active_count() > 2:
            sleep(0.1)
    except KeyboardInterrupt:
        print('Caught keyboard interrupt - shutting down.')
    finally:
        # Clean up
        try:
            keyboard.unhook_all_hotkeys()
        except (AttributeError, NameError):
            # Ignore keyboard library errors
            pass

def parse_arguments():
    """
    Parse command line arguments
    
    Returns:
        args: Parsed arguments
    """
    parser = ArgumentParser(description='Pig Chase DDQN Experiment')
    
    # Backend selection
    parser.add_argument(
        '-b', '--backend',
        type=str,
        choices=['cntk', 'chainer', 'pytorch'],
        #default='cntk',
        default='pytorch',
        help='Neural network backend'
    )
    
    # Training duration
    parser.add_argument(
        '-e', '--epochs',
        type=int,
        default=5,
        help='Number of epochs to run'
    )
    
    # Minecraft clients
    parser.add_argument(
        'clients',
        nargs='*',
        default=DEFAULT_CLIENTS,
        help='Minecraft client endpoints (ip:port)'
    )
    
    # Hardware device
    parser.add_argument(
        '-d', '--device',
        type=int,
        default=-1,
        help='GPU device index (-1 for CPU)'
    )
    
    # Simulation speed
    parser.add_argument(
        '-hs', '--human-speed',
        action='store_true',
        help='Run at human-observable speed'
    )
    
    # Pre-trained model
    parser.add_argument(
        '-m', '--model-file',
        type=str,
        default=None,
        help='Path to pre-trained model to load'
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the Pig Chase DDQN experiment"""
    # Parse command-line arguments
    args = parse_arguments()
    
    # Create timestamped log directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    logdir = os.path.join(RESULTS_FOLDER, timestamp)
    os.makedirs(logdir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = create_visualizer(logdir)
    
    # Create agent definitions
    agents = []
    print('clients:',parse_clients_args(args.clients))
    for role, agent_name in enumerate(ENV_AGENT_NAMES):
        agent_def = {
            'role': role,
            'name': agent_name,
            'clients': parse_clients_args(args.clients),
            'backend': args.backend,
            'device': args.device,
            'max_epochs': args.epochs,
            'logdir': logdir,
            'visualizer': visualizer,
            'human_speed': args.human_speed,
            'model_file': args.model_file
        }
        print('agent_def:', agent_def)
        agents.append(agent_def)
    
    print(f'Starting Pig Chase DDQN experiment in {logdir}')
    
    # Start the experiment
    run_experiment(agents)
    
    print('Experiment complete')

if __name__ == '__main__':
    main()


