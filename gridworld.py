"""
An implementation of the grid world experiment from:

Wilson et al. "Multi-Task Reinforcement Learning: A 
Hierarchical Bayesian Approach." Proceedings of the 24th International
Conference on Machine Learning (ICML'07). ACM, 2007.

During task:
r ~ N(r | mu, tau)
mu = w . Q

Task creation:
w ~ N(w | phi, Sigma)
Q ~ Un(colors)
pi(phi, sigma) = NIW

Experiment initialization:
theta_c ~ multinomial(classes)
theta_c = (phi_c, Sigma_c)

To explore:
- Simultaneous tasks
- Gaps in observations
- Shared parameters across all MDPs
- Each MPD composed of multiple classes
"""
import random
import math
import numpy as np

CURRENT = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4

NUM_ACTIONS = 4
NUM_RELATIVE_CELLS = 5

RELATIVE_CELL = ['C', 'U', 'D', 'L', 'R']
ACTION_NAMES = ['U', 'D', 'L', 'R']


class Agent(object):
    """
    An abstract base class that grid world agents must extend.
    """
    def __init__(self, width, height, num_colors, num_domains, name=None):
        self.name = name
        self.width = width
        self.height = height
        self.colors = num_colors
        self.domains = [None] * num_domains
        self.state = [None] * num_domains
        self.location = [None] * num_domains
        self.domain_episodes = np.zeros((num_domains))
        self.total_episodes = 0
        self.recent_rewards = []

    def episode_starting(self, idx, location, state):
        self.location[idx] = location
        self.state[idx] = state

    def episode_over(self, idx):
        self.total_episodes += 1
        self.domain_episodes[idx] += 1

    def get_action(self, idx):
        pass

    def set_state(self, idx, location, state):
        self.location[idx] = location
        self.state[idx] = state

    def observe_reward(self, idx, r):
        self.recent_rewards.append(r)
        if len(self.recent_rewards) > 10:
            self.recent_rewards.pop(0)

class GridWorld(object):
    def __init__(self, task_id, color_location_weights, reward_stdev = 2, agent = None, width = 15, height = 15, max_moves = 100, start = (0,0), goal = None):
        self.task_id = task_id
        self.color_location_weights = color_location_weights
        # We need one weight for every (loc, color) pair
        assert(len(color_location_weights) % 5 == 0)
        self.num_colors = len(color_location_weights) / 5
        self.reward_stdev = reward_stdev
        self.agent = agent
        self.width = width
        self.height = height
        self.max_moves = max_moves
        self.build_cells()
        self.start_location = start
        if goal is None:
            goal = (width-1,height-1)
        self.goal = goal
        self.episode_running = False
        self.state = None
        self.location = None

    def build_cells(self):
        self.cell_colors = np.array([[random.randrange(self.num_colors) for y in range(self.height)] for x in range(self.width)])
        self.cell_means = np.zeros((self.width, self.height))
        self.cell_states = np.zeros((self.width, self.height, len(self.color_location_weights)))
        for x in range(self.width):
            for y in range(self.height):
                self.cell_states *= 0 # zero out the weights
                self.cell_states[x,y,CURRENT*self.num_colors + self.cell_colors[x,y]] = 1
                if y > 0:
                    self.cell_states[x,y,UP*self.num_colors + self.cell_colors[x,y-1]] = 1
                if y < (self.height - 1):
                    self.cell_states[x,y,DOWN*self.num_colors + self.cell_colors[x,y+1]] = 1
                if x > 0:
                    self.cell_states[x,y,LEFT*self.num_colors + self.cell_colors[x-1,y]] = 1
                if x < (self.width - 1):
                    self.cell_states[x,y,RIGHT*self.num_colors + self.cell_colors[x+1,y]] = 1
                # mu = w . Q
                self.cell_means[x,y] = np.dot(self.color_location_weights, self.cell_states[x,y])

    def start(self):
        self.prev_location = None
        self.location = self.start_location
        self.state = self.cell_states[self.location]
        self.total_reward = 0
        self.episode_running = True
        self.agent.episode_starting(self.task_id, self.location, self.state)

    def reward(self, action):
        return random.normalvariate(self.cell_means[self.location], self.reward_stdev)

    def transition(self, action):
        """
        Transition function given an action. The default grid world uses a deterministic
        transition that simply moves the agent where it wants to go, unless it hits a wall.
        """
        self.prev_location = self.location
        if action == UP:
            self.location = (self.location[0], max(0, self.location[1]-1))
        elif action == DOWN:
            self.location = (self.location[0], min(self.height - 1, self.location[1] + 1))
        elif action == LEFT:
            self.location = (max(0, self.location[0] - 1), self.location[1])
        elif action == RIGHT:
            self.location = (min(self.width-1, self.location[0] + 1), self.location[1])
        self.state = self.cell_states[self.location]
        
    def step(self):
        assert(self.episode_running)
        action = self.agent.get_action(self.task_id)
        self.transition(action)
        r = self.reward(action)
        self.total_reward += r
        self.agent.observe_reward(self.task_id, r)
        self.agent.set_state(self.task_id, self.location, self.state)
        if self.location == self.goal:
            self.agent.episode_over(self.task_id)
            self.episode_running = False

    def play_episode(self):
        self.start()
        for i in range(self.max_moves):
            self.step()
            if not self.episode_running:
                break
        return self.total_reward

    def print_world(self, cell_values=None):
        if cell_values is None:
            cell_values = self.cell_means
        reward_format = '{0}'
        if cell_values.dtype is np.dtype(float):
            reward_format = '{0:.2f}'
        cell_width = 11
        side_width = int(math.floor(cell_width/2))
        side_space = ' '*side_width
        print 'Color Weights:'
        for i,row in enumerate(self.color_location_weights.reshape(5, self.num_colors)):
            print '{0}: {1}'.format(RELATIVE_CELL[i], row)
        print '-' * ((cell_width+1)*self.width+1)
        for y in range(self.height):
            if y != self.goal[1]:
                top = '|' + (side_space + '^' + side_space + '|') * self.width
                bottom = '|' + (side_space + 'v' + side_space + '|') * self.width
            else:
                top = '|' + (side_space + '^' + side_space + '|') * (self.width-1) + ' ' * cell_width + '|'
                bottom = '|' + (side_space + 'v' + side_space + '|') * (self.width-1) + ' ' * cell_width + '|'
            blank = '|' + (' ' * cell_width + '|') * self.width
            state_line = '|'
            reward_line = '|'
            color_line = '|'
            for x in range(self.width):
                state_text = ''
                if (x,y) == self.goal:
                    state_text += '**'
                    reward_line += reward_format.format(cell_values[x,y]).center(cell_width) + '|'
                else:
                    reward_line += '<' + reward_format.format(cell_values[x,y]).center(cell_width-2) + '>|'
                if self.agent is not None and self.location == (x,y):
                    state_text += 'X'
                state_line += state_text.center(cell_width) + '|'
                color_line += 'C={0}'.format(self.cell_colors[x,y]).center(cell_width) + '|'
            print top
            print state_line
            print reward_line
            print color_line
            print bottom
            print '-' * ((cell_width+1)*self.width+1)

if __name__ == "__main__":
    agent = Agent([])
    colors = ['red', 'green', 'blue', 'gray']
    means = np.random.rand(len(colors) * 5) * -2
    cov = np.random.rand(len(colors) * 5, len(colors) * 5) * 2. - 1.
    w = np.random.multivariate_normal(means, cov)
    world = GridWorld(0, w, 0.1, agent, 10, 10, 100, (0,0), None)
    world.start()
    world.print_world()