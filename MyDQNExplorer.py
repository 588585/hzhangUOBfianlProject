# to do : 计划把LinearEpsilonGreedyExplorerr改成MyDQNExplorer
# 基于UCB（Upper Confidence Bound）算法的探索器。 `UCBExplorer` 类实现了基于UCB算法的探索器。它通过以下步骤选择动作：

# 1. **初始化**：在第一次调用时初始化动作计数和动作值。
# 2. **更新**：在每次选择动作后，根据获得的奖励更新动作值。
# 3. **探索**：根据UCB公式计算每个动作的UCB值，并选择具有最高UCB值的动作。

from __future__ import absolute_import
from malmopy.agent import BaseExplorer
import numpy as np
import math

# 上置信界（Upper Confidence Bound, UCB）探索策略
#  At = argmax_a(Qt(a) + c * sqrt(ln(N) / Na)
#  其中：
#  - At 是在时间 t 选择的动作
#  - Qt(a) 是动作 a 的平均奖励
#  - c 是一个探索参数
#  - N 是总的动作选择次数
#  - Na 是动作 a 的选择次数

class BaseExplorer:
    """ Explore/exploit logic wrapper"""

    def __call__(self, step, nb_actions):
        return self.explore(step, nb_actions)

    def is_exploring(self, step):
        """ Returns True when exploring, False when exploiting """
        raise NotImplementedError()

    def explore(self, step, nb_actions):
        """ Generate an exploratory action """
        raise NotImplementedError()

class UCBExplorer(BaseExplorer):
    """ Upper Confidence Bound (UCB) Explorer """

    def __init__(self, c=1.0): 
        self.c = c 
        self.counts = None 
        self.values = None

    def initialize(self, nb_actions): 
        self.counts = np.zeros(nb_actions) # 初始化所有动作的选择次数为 0
        self.values = np.zeros(nb_actions) # 初始化所有动作的奖励值为 0

    def update(self, action, reward):  
        self.counts[action] += 1 # 增加该动作的选择次数
        n = self.counts[action] # 获取该动作的选择次数
        value = self.values[action] # 获取该动作的奖励值
        new_value = ((n - 1) / n) * value + (1 / n) * reward # 更新该动作的奖励值
        self.values[action] = new_value # 保存更新后的奖励值

        #均值更新公式：Q(a) = (N-1)/N * Q(a) + 1/N * R(a)

    def is_exploring(self, step): 
        return True  # UCB always explores

    def explore(self, step, nb_actions): 
        if self.counts is None or self.values is None: 
            self.initialize(nb_actions) 

        total_counts = np.sum(self.counts)
        if total_counts < nb_actions: # 如果总的选择次数小于动作数，则依次选择每个动作
            return np.argmin(self.counts)

        ucb_values = self.values + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-5)) # 计算每个动作的UCB值
        return np.argmax(ucb_values) # 返回UCB值最大的动作
    


class QuadraticEpsilonGreedyExplorer(BaseExplorer):
    """使用二次函数衰减的ε-贪婪探索策略
    
    ε = ε_max - (ε_max - ε_min) * (step / eps_min_time)²
    """

    def __init__(self, eps_max, eps_min, eps_min_time, before_training_steps=0):
        assert eps_max > eps_min
        assert eps_min_time > 0
        
        self._eps_min_time = eps_min_time
        self._eps_min = eps_min
        self._eps_max = eps_max
        self.before_training_steps = before_training_steps

    def _epsilon(self, step):
        if step < 0:
            return self._eps_max
        elif step > self._eps_min_time:
            return self._eps_min
        else:
            normalized_step = step / self._eps_min_time
            return self._eps_max - (self._eps_max - self._eps_min) * (normalized_step ** 2)

    def is_exploring(self, step):
        _step_after_before_training = step - self.before_training_steps
        if  _step_after_before_training < 0:
            return True
        else:
            return np.random.rand() < self._epsilon(_step_after_before_training)

    def explore(self, step, nb_actions):
        return np.random.randint(0, nb_actions)
    
if __name__ == '__main__':
    explorer  = QuadraticEpsilonGreedyExplorer(1, 0.1,320000, 50000)
    print(explorer.is_exploring(0))
    print(explorer.is_exploring(50000))
    print(explorer.is_exploring(100000))
    print(explorer.is_exploring(200000))
    print(explorer.is_exploring(300000))
    print(explorer.is_exploring(400000))
    print(explorer.is_exploring(500000))
    print(explorer.is_exploring(600000))
    print(explorer.is_exploring(700000))