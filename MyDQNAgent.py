from __future__ import absolute_import
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import numpy
import cntk as C
import pickle
from cntk import CloneMethod
import copy


# from chainer.serializers import save_npz, load_npz

from collections import deque, namedtuple
from common import ENV_CAUGHT_REWARD

from malmopy.agent import BaseAgent, ReplayMemory, BaseExplorer, LinearEpsilonGreedyExplorer, TemporalMemory
from malmopy.model.cntk import QNeuralNetwork 

from malmopy.model import QModel
from malmopy.util import get_rank
#from MyDQNExplorer import QuadraticEpsilonGreedyExplorer , BaseExplorer

# Track previous state and action for observation 。 创建Tracker命名元组，用于存储 先前的状态和动作，方便训练时计算 Q-learning 目标值。
Tracker = namedtuple('Tracker', ['state', 'action'])

_allow_pickle_kwargs = {}
ms_actions = 3
ms_memory = TemporalMemory(100000, (84, 84))
ms_model = QNeuralNetwork((ms_memory.history_length, 84, 84), 3,-1)

            
class MyDDQN_Q(BaseAgent):


    def __init__(self, name, actions= ms_actions , model = ms_model, memory = ms_memory, gamma=0.99, batch_size=32, learning_rate=1e-3, target_network_train_frequency= 50000, # 目标网络更新频率已经弃用，实际上使用的是my_train_after
                 explorer=None, visualizer=None,backend='cntk', My_train_after=10000, reward_clipping=None, is_evaluating=False, train_frequency=8, tau=1.0):
        
        print('name:',name,'actions:',actions,'model:',model,'memory:',memory,'gamma:',gamma,
              'batch_size:',batch_size,'learning_rate:',learning_rate,'target_network_train_frequency:',target_network_train_frequency,
              'explorer:',explorer,'visualizer:',visualizer,'backend:',backend,'My_train_after:',My_train_after)


        train_frequency = train_frequency #
        assert isinstance(model, QModel), 'model should inherit from QModel'
        # assert isinstance(model, nn.Module), 'model should be a PyTorch nn.Module'
        assert get_rank(model.input_shape) > 1, 'input_shape rank should be > 1'
        assert isinstance(memory, ReplayMemory), 'memory should inherit from ' \
                                                 'ReplayMemory'
        assert 0 < gamma < 1, 'gamma should be 0 < gamma < 1 (got: %d)' % gamma
        assert batch_size > 0, 'minibatch_size should be > 0 (got: %d)' % batch_size
        assert My_train_after >= 0, 'train_after should be >= 0 (got %d)' % My_train_after
        assert train_frequency > 0, 'train_frequency should be > 0'

        super(MyDDQN_Q, self).__init__(name, actions, visualizer)
        
        self._actions = actions  # 
        self._model = model  # 
        self._memory = memory  # 
        self._gamma = gamma  # 
        self._batch_size = batch_size # minibatch_size
        #self._learning_rate = learning_rate# 
        self._tracker = None# 
        self._train_after = My_train_after# 
        self._actions_taken = 0 # 
        self._history = History(model.input_shape) # 
        self._train_frequency = train_frequency # 
        self._is_evaluating = is_evaluating #
        #self._target_network = copy.deepcopy(self._model) 
        self._target_network = QNeuralNetwork((ms_memory.history_length, 84, 84), 3,-1)
        #self._tau = 0.005  # 软更新系数
        self.update_target_network_count=0

        reward_clipping = reward_clipping or (-2 ** 31 - 1, 2 ** 31 - 1) 
        assert isinstance(reward_clipping, tuple) and len(reward_clipping) == 2,'clip_reward should be None or (min_reward, max_reward)'
        assert reward_clipping[0] <= reward_clipping[1],'max reward_clipping should be >= min (got %d < %d)' % (
                reward_clipping[1], reward_clipping[0])

        self._reward_clipping = reward_clipping

        explorer = explorer or LinearEpsilonGreedyExplorer(1, 0.1, 1e6)
        assert isinstance(explorer, BaseExplorer), 'explorer should inherit from BaseExplorer'
        self._explorer = explorer

        # Stats related
        self._stats_rewards = [] 
        self._stats_mean_qvalues = [] 
        self._stats_stddev_qvalues = [] 
        self._stats_loss = [] 


    def change_model(self, model):
        self._model = model

    def act(self, state, reward, done, is_training=True):
        new_state = state

        if self._tracker is not None:
            self.observe(self._tracker.state, self._tracker.action,reward, new_state, done)

        if is_training: 
            #print('self._train_after:',self._train_after)
            #print('self._actions_taken:',self._actions_taken)
            if self._actions_taken >  self._train_after : 

                self.learn()
        # Append the new state to the history
        self._history.append(new_state)


        # select the next action
        if self._explorer.is_exploring(self._actions_taken) and not( self._is_evaluating ):
            new_action = self._explorer(self._actions_taken, self.nb_actions)

        else:
            q_values = self._model.evaluate(self._history.value)
            new_action = q_values.argmax()
            target_q_values = self._target_network.evaluate(self._history.value)
            target_action = target_q_values.argmax()
            #if 0 <= self._actions_taken % 500 <= 20:
            print('q_values:', q_values, 'target_q_values:', target_q_values)
            #new_action = target_action
            #print('q_values:',q_values,'new_action:',new_action,'actions:', self._actions)
            self._stats_mean_qvalues.append(q_values.max())     
            self._stats_stddev_qvalues.append(np.std(q_values)) 

        self._tracker = Tracker(new_state, new_action)
        self._actions_taken += 1 #
        return new_action 

    def observe(self, old_state, action, reward, new_state, is_terminal):  
        if is_terminal:
            self._history.reset()

        min_val, max_val = self._reward_clipping #
        reward = max(min_val, min(max_val, reward)) 
        self._memory.append(old_state, int(action), reward, is_terminal) 


    def learn(self): 

        if (self._actions_taken % self._train_frequency) == 0 :
            minibatch = self._memory.minibatch(self._batch_size) 
            q_in_learn = self._compute_q(*minibatch)   

            self._model.train(minibatch[0], q_in_learn , minibatch[1]) 
            self._stats_loss.append(self._model.loss_val)
        self.update_target_network()
        
    def update_target_network(self): 
        """ Perform a soft update of target network """
        if self._actions_taken % 1000== 0 : 
            try:
                for target_param, source_param in zip(
                        self._target_network._model.parameters, 
                        self._model._model.parameters):
                    target_param.value = source_param.value.copy()
                print("目标网络参数更新成功")
            except Exception as e:
                print(f"目标网络更新失败: {e}")

    def inject_summaries(self, idx):  # 。
        if len(self._stats_mean_qvalues) > 0: #
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

            # Reset
            self._stats_mean_qvalues = []
            self._stats_stddev_qvalues = []
            self._stats_loss = []
            self._stats_rewards = []

    def _compute_q(self, pres, actions, posts, rewards, terminals):
        """ Compute the Q Values from input states using DDQN """

        q_eval = self._model.evaluate(posts)  # Q(s', a; θ)
        best_actions = q_eval.argmax(axis=1)

        q_hat = self._target_network.evaluate(posts, model=QModel.TARGET_NETWORK)  # Q_target(s', a')
        q_hat_eval = q_hat[np.arange(len(actions)), best_actions]  # Q_target(s', argmax_a Q(s',a))

        q_targets = rewards + (1 - terminals) * (self._gamma * q_hat_eval)  
        return np.array(q_targets, dtype=np.float32)



    def store_transition(self,old_state, action, reward, is_terminal): 

        self._memory.append(old_state, action, reward, is_terminal)
  

class History(object):
 

    def __init__(self, shape):
        self._buffer = np.zeros(shape, dtype=np.float32) 

    @property 
    def value(self):   
        return self._buffer

    def append(self, state): 
        self._buffer[:-1] = self._buffer[1:] 
        self._buffer[-1, ...] = state

    def reset(self): 
        self._buffer.fill(0)
        # print('reset history buffer')

class TemporalMemory(ReplayMemory): 
    def __init__(self, max_size, sample_shape, history_length=4, unflicker=False): #

        super(TemporalMemory, self).__init__(max_size, sample_shape) #

        self._unflicker = unflicker
        self._history_length = max(1, history_length)
        self._last = np.zeros(sample_shape)  

    def append(self, state, action, reward, is_terminal): 
        if self._unflicker: 
            max_diff_buffer = np.maximum(self._last, state)
            self._last = state
            state = max_diff_buffer

        super(TemporalMemory, self).append(state, action, reward, is_terminal) 
        if is_terminal: 
            if self._unflicker:
                self._last.fill(0) 




    def sample(self, size, replace=True): 

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

        return self._unflicker

    @property
    def history_length(self):

        return self._history_length
    
