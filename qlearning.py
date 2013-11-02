import random
from itertools import product
from gridworld import *

class QAgent(Agent):
    """
    A Q-Learning agent with optimistic initialization.
    """
    def __init__(self, width, height, colors, num_domains, name=None, epsilon = 0.1, alpha = 0.05, gamma = 1.):
        Agent.__init__(self, width, height, colors, num_domains, name)
        assert(epsilon >= 0.)
        assert(epsilon <= 1.)
        assert(alpha >= 0.)
        assert(alpha <= 1.)
        assert(gamma >= 0.)
        assert(gamma <= 1.)
        self.epsilon = epsilon # Exploration rate
        self.alpha = alpha # Learning rate
        self.gamma = gamma # Discount factor
        # Initialize the state-action tables
        self.q = [np.zeros((width, height, NUM_ACTIONS)) for domain in range(num_domains)]
        self.visits = [np.zeros((width, height)) for domain in range(num_domains)]
        self.q_visits = [np.zeros((width, height, NUM_ACTIONS)) for domain in range(num_domains)]
        self.update = [False for domain in range(num_domains)]
        self.prev_state = [None for domain in range(num_domains)]
        self.prev_action = [None for domain in range(num_domains)]
        self.prev_reward = [None for domain in range(num_domains)]
        
    def episode_starting(self, idx, state):
        self.update[idx] = False
        self.prev_state[idx] = None
        super(QAgent, self).episode_starting(idx, state)

    def episode_over(self, idx):
        if self.update[idx]:
            qidx = (self.prev_state[idx][0], self.prev_state[idx][1], self.prev_action[idx])
            self.q[idx][qidx] += self.alpha * (self.prev_reward[idx] - self.q[idx][qidx])
        super(QAgent, self).episode_over(idx)

    def get_action(self, idx):
        if self.update[idx]:
            self.update_q(idx)
        else:
            self.update[idx] = True
        if random.random() < self.epsilon:
            action = random.randrange(NUM_ACTIONS)
        else:
            (action, val) = self.greedy(idx)
        self.prev_action[idx] = action
        self.q_visits[idx][self.state[idx][0],self.state[idx][1],action] += 1
        return action+1

    def greedy(self, idx, state=None, debug=False):
        maxi = []
        maxv = 0
        if state is None:
        	state = self.state[idx]
        for i in range(NUM_ACTIONS):
            action_val = self.q[idx][state[0],state[1],i]
            if debug:
                print '\tQ[{0},{1}] = {2}'.format(state, ACTION_NAMES[i], action_val)
            if len(maxi) == 0 or action_val > maxv:
                maxi = [i]
                maxv = action_val
            elif action_val == maxv:
                maxi.append(i)
        if len(maxi) == 1:
            maxi = maxi[0]
        else:
            maxi = random.choice(maxi)
        if debug:
            print '\tChoosing: {0}'.format(ACTION_NAMES[maxi])
        return (maxi,maxv)

    def update_q(self, idx):
        prev_qidx = (self.prev_state[idx][0],self.prev_state[idx][1],self.prev_action[idx])
        self.q[idx][prev_qidx] += self.alpha * (self.prev_reward[idx] + self.gamma * self.greedy(idx)[1] - self.q[idx][prev_qidx])

    def set_state(self, idx, state):
        self.prev_state[idx] = self.state[idx]
        super(QAgent, self).set_state(idx, state)
        self.visits[idx][state] += 1

    def observe_reward(self, idx, r):
        self.prev_reward[idx] = r
        super(QAgent, self).observe_reward(idx, r)

    def get_policy(self, idx):
        domain = self.domains[idx]
        pi = np.array([[self.greedy(idx,state=(x,y))[0]+1 for y in range(domain.height)] for x in range(domain.width)])
        values = np.array([[self.greedy(idx,state=(x,y))[1] for y in range(domain.height)] for x in range(domain.width)])
        return (pi, values)

if __name__ == "__main__":
    width = 10
    height = 10
    colors = 4
    domains = 2
    agent = QAgent(width, height, colors, domains, name='Q-Learning Agent', epsilon=0.1, alpha=0.05, gamma=1.)
    means = np.random.rand(colors * 5) * -10.
    cov = np.random.rand(colors * 5, colors * 5) * 2. - 1.
    w = np.random.multivariate_normal(means, cov)
    world = GridWorld(0, w, 0.1, None, 10, 10, 100, (0,0), None)
    world2 = GridWorld(0, w, 0.1, None, 10, 10, 100, (0,0), None)
    world.agent = agent
    agent.domains = [world, world2]
    world.start()
    world.print_world()
    for i in range(1000):
    	world.play_episode()
    (pi, values) = agent.get_policy(0)
    print 'VALUES'
    world.print_world(values)
    print 'POLICY'
    names = np.array(ACTION_NAMES)[pi-1]
    world.print_world(names)