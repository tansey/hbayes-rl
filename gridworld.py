"""
An implementation of the grid world experiment from:

Wilson, Aaron, et al. "Multi-Task Reinforcement Learning: A 
Hierarchical Bayesian Approach." Proceedings of the 24th International
Conference on Machine Learning (ICML'07). ACM, 2007.

To explore:
- Simultaneous tasks
- Gaps in observations
- Shared parameters across all MDPs
- Each MPD composed of multiple classes
"""
import random
import math
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
    def __init__(self, task_id, agent = None, width = 15, height = 15, max_moves = 100, color_scores = (-1), start = (0,0), goal = None):
        self.task_id = task_id
        self.agent = agent
        self.width = width
        self.height = height
        self.max_moves = max_moves
        self.color_scores = color_scores
        self.build_cells()
        self.start_state = start
        if goal is None:
            goal = (width-1,height-1)
        self.goal = goal
        self.episode_running = False
        self.state = None

    def build_cells(self):
        self.cell_colors = np.array([[random.randrange(len(self.color_scores)) for y in range(self.height)] for x in range(self.width)])
        self.cell_values = np.array([[self.color_scores[self.cell_colors[x,y]] for y in range(self.height)] for x in range(self.width)])
        for x in range(self.width):
            for y in range(self.height):
                if x > 0:
                    self.cell_values[x][y] += self.color_scores[self.cell_colors[x-1,y]]
                if x < (self.width - 1):
                    self.cell_values[x][y] += self.color_scores[self.cell_colors[x+1,y]]
                if y > 0:
                    self.cell_values[x][y] += self.color_scores[self.cell_colors[x,y-1]]
                if y < (self.height - 1):
                    self.cell_values[x][y] += self.color_scores[self.cell_colors[x,y+1]]

    def start(self):
        self.prev_state = None
        self.state = self.start_state
        self.total_reward = 0
        self.episode_running = True
        self.agent.episode_starting(self.task_id, self.state)

    def reward(self, action):
        return self.cell_values[self.state[0], self.state[1]]

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
        if self.state == self.goal:
            self.agent.episode_over(self.task_id)
            self.episode_running = False

    def play_episode(self):
        self.start()
        for i in range(self.max_moves):
            self.step()
            if not self.episode_running:
                break
        return total_reward

    def print_world(self):
        cell_width = 11
        side_width = int(math.floor(cell_width/2))
        side_space = ' '*side_width
        print '{0} Colors'.format(len(self.color_scores))
        for color,score in enumerate(self.color_scores):
            print '{0}) value = {1}'.format(color, score)
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
                    reward_line += '{0:.2f}'.format(self.cell_values[x,y]).center(cell_width) + '|'
                else:
                    reward_line += '<' + '{0:.2f}'.format(self.cell_values[x,y]).center(cell_width-2) + '>|'
                if self.agent is not None and self.state == (x,y):
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
    agent = Agent(None)
    color_means = (-4,-5,-2,-3)
    # TODO: What should the variance be? The paper does not specify values here.
    color_scores = np.array([random.normalvariate(mu, 1) for mu in color_means])
    world = GridWorld(0, agent, 10, 10, 100, color_scores, (0,0), None)
    world.start()
    world.print_world()