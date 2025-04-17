from __future__ import absolute_import
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import copy

from collections import deque, namedtuple
from common import ENV_CAUGHT_REWARD

from malmopy.agent import BaseAgent, ReplayMemory, BaseExplorer, LinearEpsilonGreedyExplorer, TemporalMemory
from malmopy.model import QModel
from malmopy.util import get_rank



# Track previous state and action for observation
Tracker = namedtuple('Tracker', ['state', 'action'])

class PyTorchQModel(QModel):
    """PyTorch implementation of a Q-Network model"""
    
    # Class variables
    loss_val = 0  # Store the last loss value
    
    def __init__(self, input_shape, output_shape, device=None, tau=1.0):
        """
        Initialize a PyTorch Q-Network model
        
        Args:
            input_shape: Shape of the input observations
            output_shape: Number of possible actions
            device: Device to run the model on ('cpu' or 'cuda')
            tau: Soft update parameter for target network
        """
        super(PyTorchQModel, self).__init__(input_shape, output_shape)
        
        self._nb_actions = output_shape
        self._tau = tau

        if device is None:
            # Auto-select device
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            # Use specified device or fallback to CPU
            if isinstance(device, int):
                if device >= 0 and torch.cuda.is_available() and device < torch.cuda.device_count():
                    self.device = f'cuda:{device}'
                else:
                    print(f"Warning: GPU device {device} not available, using CPU instead")
                    self.device = 'cpu'
            else:
                # Invalid device identifier
                print(f"Warning: Device identifier '{device}' is invalid, using CPU instead")
                self.device = 'cpu'
                
        print(f"Using device: {self.device}")
        
        # Build the network
        self._model = self._build_model().to(self.device)
        self._target_network = self._build_model().to(self.device)
        
        # Copy weights to target network
        self._update_target_network(tau=1.0)
        
        # Setup optimizer
        self.optimizer = optim.Adam(self._model.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss
    
    def _build_model(self):
        """
        Build the neural network model
        
        Returns:
            A PyTorch neural network model with convolutional and fully connected layers
        """
        model = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, self._nb_actions)
        )
        return model
    
    def _update_target_network(self, tau=None):
        """
        Update the target network parameters
        
        Args:
            tau: Soft update parameter (1.0 for hard update)
        """
        tau = tau if tau is not None else self._tau
        
        for target_param, param in zip(self._target_network.parameters(), self._model.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    
    def train(self, states, q_targets, actions):
        """
        Train the model with a batch of experiences
        
        Args:
            states: Batch of states
            q_targets: Target Q-values
            actions: Actions taken in each state
        """
        states = torch.FloatTensor(states).to(self.device)
        q_targets = torch.FloatTensor(q_targets).to(self.device)
        
        # Convert actions to PyTorch tensor
        actions_tensor = torch.LongTensor(actions).to(self.device)
        
        # Forward pass to get all Q-values
        q_values = self._model(states)
        
        # Use gather to select Q-values for corresponding actions
        q_values_for_actions = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        
        # Calculate loss and update
        loss = self.loss_fn(q_values_for_actions, q_targets)
        PyTorchQModel.loss_val = loss.item()  # Update class variable
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def evaluate(self, data, model=QModel.ACTION_VALUE_NETWORK):
        """
        Evaluate the model on input data
        
        Args:
            data: Input data
            model: Which network to use (ACTION_VALUE_NETWORK or TARGET_NETWORK)
            
        Returns:
            Q-values for each action
        """
        # Ensure data is the correct type
        if not isinstance(data, torch.Tensor):
            # Convert to torch tensor
            data = torch.FloatTensor(data).to(self.device)
        
        # Ensure the input has the right shape
        if len(data.shape) == len(self.input_shape):
            data = data.unsqueeze(0)  # Add batch dimension
        
        with torch.no_grad():
            if model == QModel.TARGET_NETWORK:
                q_values = self._target_network(data)
            else:
                q_values = self._model(data)
                
        # Ensure returning a 1D array if only one sample
        result = q_values.cpu().numpy()
        if result.shape[0] == 1:
            return result.squeeze(0)  # Remove batch dimension, return 1D array
        return result
    
    def save(self, filepath):
        """
        Save the model weights to a file
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'model_state_dict': self._model.state_dict(),
            'target_state_dict': self._target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, filepath)
        print(f"✅ Model saved to: {filepath}")
    
    def load(self, filepath):
        """
        Load the model weights from a file
        
        Args:
            filepath: Path to the saved model
        """
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._target_network.load_state_dict(checkpoint['target_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"✅ Model loaded from: {filepath}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            raise

class MyDDQN_Q(BaseAgent):
    """Double Deep Q-Network agent implementation with PyTorch"""

    def __init__(self, name, actions, model, memory, gamma=0.99, batch_size=32, 
                 learning_rate = 0, target_network_train_frequency=1000,
                 explorer=None, visualizer=None, backend='pytorch', My_train_after=10000, 
                 reward_clipping=None, is_evaluating=False, train_frequency=8, tau=1.0):
        """
        Initialize DDQN agent
        
        Args:
            name: Agent name
            actions: List of available actions
            model: Q-Network model
            memory: Experience replay buffer
            gamma: Discount factor
            batch_size: Batch size for training
            learning_rate: !!! not used in this implementation 
            target_network_train_frequency: Frequency of target network updates
            explorer: Exploration strategy
            visualizer: Visualization tool
            backend: Neural network backend
            My_train_after: Number of steps before training starts
            reward_clipping: Range for reward clipping
            is_evaluating: Whether to run in evaluation mode
            train_frequency: How often to train the network
            tau: Soft update parameter for target network
        """
        
        print('name:', name, 'actions:', actions, 'model:', model, 'memory:', memory, 'gamma:', gamma,
              'batch_size:', batch_size,
              'explorer:', explorer, 'visualizer:', visualizer, 'backend:', backend, 
              'My_train_after:', My_train_after)

        train_frequency = 8  # Training frequency
        assert isinstance(model, QModel), 'model should inherit from QModel'
        assert get_rank(model.input_shape) > 1, 'input_shape rank should be > 1'
        assert isinstance(memory, ReplayMemory), 'memory should inherit from ReplayMemory'
        assert 0 < gamma < 1, 'gamma should be 0 < gamma < 1 (got: %d)' % gamma
        assert batch_size > 0, 'minibatch_size should be > 0 (got: %d)' % batch_size
        assert My_train_after >= 0, 'train_after should be >= 0 (got %d)' % My_train_after
        assert train_frequency > 0, 'train_frequency should be > 0'


        super(MyDDQN_Q, self).__init__(name, actions, visualizer)
        
        self._actions = actions  # Action space
        self._model = model  # PyTorch QModel
        self._memory = memory  # Experience replay buffer
        self._gamma = gamma  # Discount factor
        self._batch_size = batch_size  # Minibatch size
        self._tracker = None  # Tracker for previous state and action
        self._train_after = My_train_after  # Number of actions before training starts
        self._actions_taken = 0  # Number of actions taken
        self._history = History(model.input_shape)  # State history buffer
        self._train_frequency = train_frequency  # Training frequency
        self._is_evaluating = is_evaluating  # Whether we're evaluating
        self._target_update_freq = target_network_train_frequency  # Frequency of target network updates
        self._tau = tau  # Soft update parameter for target network
        self._backend = backend  # Backend for the model
        
        reward_clipping = reward_clipping or (-2 ** 31 - 1, 2 ** 31 - 1)
        assert isinstance(reward_clipping, tuple) and len(reward_clipping) == 2, \
            'clip_reward should be None or (min_reward, max_reward)'
        assert reward_clipping[0] <= reward_clipping[1], \
            'max reward_clipping should be >= min (got %d < %d)' % (
                reward_clipping[1], reward_clipping[0])

        self._reward_clipping = reward_clipping

        explorer = explorer or LinearEpsilonGreedyExplorer(1, 0.1, 1e6)
        assert isinstance(explorer, BaseExplorer), 'explorer should inherit from BaseExplorer'
        self._explorer = explorer

        # Stats related
        self._stats_rewards = []  # Rewards
        self._stats_mean_qvalues = []  # Q-values
        self._stats_stddev_qvalues = []  # Q-value standard deviation
        self._stats_loss = []  # Loss values

    def act(self, state, reward, done, is_training=True):
        """
        Choose an action based on the current state
        
        Args:
            state: Current state
            reward: Previous reward
            done: Whether the episode is done
            is_training: Whether we're training
            
        Returns:
            Selected action
        """
        new_state = state

        if self._tracker is not None:
            self.observe(self._tracker.state, self._tracker.action, reward, new_state, done)

        if is_training:
            if self._actions_taken > self._train_after:
                self.learn()
                
        # Append the new state to the history
        self._history.append(new_state)

        # Select the next action
        if self._explorer.is_exploring(self._actions_taken) and not self._is_evaluating:
            # Exploration: random action
            new_action = self._explorer(self._actions_taken, self.nb_actions)
        else:
            # Exploitation: best action according to Q-values
            q_values = self._model.evaluate(self._history.value)
            new_action = q_values.argmax()
            
            # Get target network Q-values for debugging
            target_q_values = self._model.evaluate(self._history.value, model=QModel.TARGET_NETWORK)
            target_action = target_q_values.argmax()
            
            print('q_values:', q_values, 'target_q_values:', target_q_values)
            
            self._stats_mean_qvalues.append(q_values.max())
            self._stats_stddev_qvalues.append(np.std(q_values))

        self._tracker = Tracker(new_state, new_action)
        self._actions_taken += 1
        return new_action

    def observe(self, old_state, action, reward, new_state, is_terminal):
        """
        Store an experience in the replay buffer
        
        Args:
            old_state: Previous state
            action: Action taken
            reward: Reward received
            new_state: New state
            is_terminal: Whether this is a terminal state
        """
        if is_terminal:
            self._history.reset()

        min_val, max_val = self._reward_clipping
        reward = max(min_val, min(max_val, reward))
        self._memory.append(old_state, int(action), reward, is_terminal)

    def learn(self):
        """Train the model using a batch of experiences"""
        if (self._actions_taken % self._train_frequency) == 0:
            minibatch = self._memory.minibatch(self._batch_size)
            q_targets = self._compute_q(*minibatch)
            self._model.train(minibatch[0], q_targets, minibatch[1])
            self._stats_loss.append(self._model.loss_val)
            
        # Update target network periodically

        if self._backend == 'pytorch':  # Assuming _backend is a class attribute initialized in __init__
            if self._actions_taken % self._target_update_freq == 0:
                self._model._update_target_network(tau=1.0)  # Hard update
                print("Target network updated")
        else:
            print("back mast be pytorch")
            sys.exit(1)

    def _compute_q(self, pres, actions, posts, rewards, terminals):
        """
        Compute target Q-values using Double DQN algorithm
        
        Args:
            pres: Previous states
            actions: Actions taken
            posts: Next states
            rewards: Rewards received
            terminals: Terminal flags
            
        Returns:
            Target Q-values
        """
        # Ensure posts data has correct shape
        posts_tensor = torch.FloatTensor(posts).to(self.device) if hasattr(self, 'device') else torch.FloatTensor(posts)
        
        # Main network selects actions
        with torch.no_grad():
            q_eval = self._model.evaluate(posts)
            best_actions = q_eval.argmax(axis=1)
            
            # Target network evaluates those actions
            q_hat = self._model.evaluate(posts, model=QModel.TARGET_NETWORK)
            q_hat_eval = q_hat[np.arange(len(actions)), best_actions]
        
        # Calculate TD targets
        q_targets = rewards + (1 - terminals) * (self._gamma * q_hat_eval)
        return np.array(q_targets, dtype=np.float32)

    def inject_summaries(self, idx):
        """
        Record metrics for visualization
        
        Args:
            idx: Current step index
        """
        if len(self._stats_mean_qvalues) > 0:
            self.visualize(idx, "%s/episode mean q" % self.name,
                           np.asscalar(np.mean(self._stats_mean_qvalues)))
            self.visualize(idx, "%s/episode mean stddev.q" % self.name,
                           np.asscalar(np.mean(self._stats_stddev_qvalues)))

        if len(self._stats_loss) > 0:
            self.visualize(idx, "%s/episode mean loss" % self.name,
                           np.asscalar(np.mean(self._stats_loss)))

        if len(self._stats_rewards) > 0:
            self.visualize(idx, "%s/episode mean reward" % self.name,
                           np.asscalar(np.mean(self._stats_rewards)))

            # Reset stats
            self._stats_mean_qvalues = []
            self._stats_stddev_qvalues = []
            self._stats_loss = []
            self._stats_rewards = []


class History(object):
    """
    Accumulator keeping track of the N previous frames to be used by the agent for evaluation
    """

    def __init__(self, shape):
        """
        Initialize history buffer
        
        Args:
            shape: Shape of the history buffer
        """
        self._buffer = np.zeros(shape, dtype=np.float32)

    @property
    def value(self):
        """Get the current history buffer value"""
        return self._buffer

    def append(self, state):
        """
        Add a new state to the history buffer
        
        Args:
            state: New state to add
        """
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1, ...] = state

    def reset(self):
        """Reset the history buffer"""
        self._buffer.fill(0)


class TemporalMemory(ReplayMemory):
    """
    Temporal memory adds a new dimension to store N previous samples (t, t-1, t-2, ..., t-N) when sampling from memory
    """

    def __init__(self, max_size, sample_shape, history_length=4, unflicker=False):
        """
        Initialize temporal memory
        
        Args:
            max_size: Maximum number of elements in the memory
            sample_shape: Shape of each sample
            history_length: Length of the visual memory (n previous frames) included with each state
            unflicker: Indicate if we need to compute the difference between consecutive frames
        """
        super(TemporalMemory, self).__init__(max_size, sample_shape)

        self._unflicker = unflicker
        self._history_length = max(1, history_length)
        self._last = np.zeros(sample_shape)

    def append(self, state, action, reward, is_terminal):
        """
        Add an experience to memory
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            is_terminal: Whether this is a terminal state
        """
        if self._unflicker:
            max_diff_buffer = np.maximum(self._last, state)
            self._last = state
            state = max_diff_buffer

        super(TemporalMemory, self).append(state, action, reward, is_terminal)

        if is_terminal:
            if self._unflicker:
                self._last.fill(0)

    def sample(self, size, replace=True):
        """
        Generate a random minibatch
        
        Args:
            size: The minibatch size
            replace: Indicate if one index can appear multiple times (True), only once (False)
            
        Returns:
            Indexes of the sampled states
        """
        if not replace:
            assert (self._count - 1) - self._history_length >= size, \
                'Cannot sample %d from %d elements' % (
                    size, (self._count - 1) - self._history_length)

        # Local variable access are faster in loops
        count, pos, history_len, terminals = self._count - 1, self._pos, \
                                             self._history_length, self._terminals
        indexes = []

        while len(indexes) < size:
            index = np.random.randint(history_len, count)

            # Check if replace=False to not include same index multiple times
            if replace or index not in indexes:

                # if not wrapping over current pointer,
                # then check if there is terminal state wrapped inside
                if not (index >= pos > index - history_len):
                    if not terminals[(index - history_len):index].any():
                        indexes.append(index)

        assert len(indexes) == size
        return indexes

    def get_state(self, index):
        """
        Return the specified state with the visual memory
        
        Args:
            index: State's index
            
        Returns:
            Tensor[history_length, input_shape...]
        """
        index %= self._count
        history_length = self._history_length

        # If index > history_length, take from a slice
        if index >= history_length:
            return self._states[(index - (history_length - 1)):index + 1, ...]
        else:
            indexes = np.arange(index - self._history_length + 1, index + 1)
            return self._states.take(indexes, mode='wrap', axis=0)

    @property
    def unflicker(self):
        """
        Indicates whether samples added to the replay memory are preprocessed
        by taking the maximum between the current frame and the previous frame
        
        Returns:
            True if preprocessed, False otherwise
        """
        return self._unflicker

    @property
    def history_length(self):
        """
        Visual memory length
        (i.e., the number of previous frames included for each sample)
        
        Returns:
            Integer >= 0
        """
        return self._history_length
    
