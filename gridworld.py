"""
An implementation of the grid world experiment from:

Wilson, Aaron, et al. "Multi-Task Reinforcement Learning: A 
Hierarchical Bayesian Approach." Proceedings of the 24th International
Conference on Machine Learning (ICML'07). ACM, 2007.

To explore:
- Simultaneous tasks
- Gaps in observations
- Shared parameters across all MDPs
"""

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
    def __init__(self, agent = None, width = 16, height = 16, max_moves = 100):
        self.agent = agent
        self.width = width
        self.height = height
        self.max_moves = max_moves

    def play_episode(self):
        state = START
        total_reward = 0
        self.agent.episode_starting(state)
        for i in range(self.max_moves):
            bidx = self.agent.get_bandit()
            assert(bidx >= 0)
            assert(bidx < self.num_bandits)
            bandit = self.bandits[bidx]
            action = bandit.sample()
            self.agent.observe_action(action)
            sr = REWARDS[(state, action)]
            total_reward += sr[1]
            self.agent.observe_reward(sr[1])
            state = sr[0]
            self.agent.set_state(state)
            if state == GOAL:
                break
        self.agent.episode_over()
        return total_reward

if __name__ == "__main__":
    build_rewards()
    print_world()