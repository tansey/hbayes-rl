"""
An implementation of the grid world experiment from:

Wilson, Aaron, et al. "Multi-Task Reinforcement Learning: A 
Hierarchical Bayesian Approach." Proceedings of the 24th International
Conference on Machine Learning (ICML'07). ACM, 2007.

To explore:
- Simultaneous tasks
- Gaps in observations
- Shared parameters across all MDPs
- MPDs composed of multiple classes
"""
import random
import numpy as np

UP = 1
DOWN = 2
RIGHT = 3
LEFT = 4

class Agent(object):
    """
    An abstract base class that grid world agents must extend.
    """
    def __init__(self, domains):
        self.domains = domains

    def episode_starting(self, idx, state):
        pass

    def episode_over(self, idx):
        pass

    def get_action(self, idx):
        pass

    def set_state(self, idx, state):
        self.domains[idx] = state

    def observe_reward(self, idx, r):
        pass

class GridWorld(object):
    def __init__(self, task_id, agent = None, width = 15, height = 15, max_moves = 100, num_colors = 8, start = (0,0), goals = None):
        self.task_id = task_id
        self.agent = agent
        self.width = width
        self.height = height
        self.max_moves = max_moves
        self.cells = np.array([[random.randrange(num_colors) for y in range(height)] for x in range(width)])
        self.start = start
        if goals is None:
            goals = []
        self.goals = goals
        self.episode_running = False

    def start(self):
        self.prev_state = None
        self.state = self.start
        self.total_reward = 0
        self.episode_running = True
        self.agent.episode_starting(self.task_id, self.state)

    def reward(self, action):
        pass

    def transition(self, action):
        """
        Transition function given an action. The default grid world uses a deterministic
        transition that simply moves the agent where it wants to go, unless it hits a wall.
        """
        self.prev_state = self.state
        if action == LEFT:
            self.state = (max(0, self.state[0] - 1), self.state[1])
        elif action == RIGHT:
            self.state = (min(self.width-1, self.state[0] + 1), self.state[1])
        elif action == UP:
            self.state = (self.state[0], max(0, self.state[1]-1))
        elif action == DOWN:
            self.state = (self.state[0], min(self.height-1, self.state[1]-1))
        
    def step(self):
        assert(self.episode_running)
        action = self.agent.get_action(self)
        self.transition(action)
        r = self.reward(action)
        self.total_reward += r
        self.agent.observe_reward(self.task_id, r)
        self.set_state(self.state)
        if self.state in self.goals:
            self.agent.episode_over(self.task_id)
            self.episode_running = False

    def play_episode(self):
        self.start()
        for i in range(self.max_moves):
            self.step()
            if not self.episode_running:
                break
        return total_reward

if __name__ == "__main__":
    build_rewards()
    print_world()