"""
A script to replicate experiments 2 and 3 from the Wilson et. al. (ICML'07) paper.
"""
import argparse
from gridworld import *
from qlearning import QAgent
import random
import matplotlib.pyplot as plt
import numpy as np

class MdpClass(object):
    def __init__(self, class_id, args):
        self.class_id = class_id
        self.weights_mean = (np.random.rand(args.colors * NUM_RELATIVE_CELLS)+1) * -2
        self.weights_cov = np.random.rand(args.colors * NUM_RELATIVE_CELLS, args.colors * NUM_RELATIVE_CELLS) * 2. - 1.
        self.goal_mean = (random.randrange(args.gridwidth), random.randrange(args.gridheight))
        self.goal_cov = np.array([[args.gstdev**2., 0],[0,args.gstdev**2.]]) # independent x,y

def get_agents(args):
    agents = []
    for agent in args.agents:
        if agent == 'qlearning':
            agents.append(QAgent(args.gridwidth, args.gridheight, args.colors, args.domains, 'Q-Learning',\
                            args.epsilon, args.alpha, args.gamma))
        else:
            raise Exception('Unsupported agent type: ' + agent)
    return agents

def create_domain(task_id, args, classes):
    clazz = random.choice(classes)
    w = np.random.multivariate_normal(clazz.weights_mean, clazz.weights_cov)
    goal = np.random.multivariate_normal(clazz.goal_mean, clazz.goal_cov)
    goal = (int(round(goal[0])), int(round(goal[1])))
    while goal[0] < 0 or goal[1] < 0 or goal[0] >= args.gridwidth or goal[1] >= args.gridheight:
        goal = np.random.multivariate_normal(clazz.goal_mean, clazz.goal_cov)
        goal = (int(round(goal[0])), int(round(goal[1])))
    world = GridWorld(task_id, w, args.rstdev, None, args.gridwidth, args.gridheight, args.maxmoves, goal, None)
    return world

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests an agent on a multi-task RL gridworld domain.')
    parser.add_argument('agents', nargs='+', choices=['qlearning', 'singlebayes', 'multibayes'])
    # General experiment arguments
    parser.add_argument('--domains', type=int, default=100, help='The number of MDP domains the agent will experience.')
    parser.add_argument('--classes', type=int, default=4, help='The number of classes that partition the set of MDPs.')
    # Grid World arguments
    parser.add_argument('--colors', type=int, default=8, help='The number of colors in the MDP. Each cell is one color.')
    parser.add_argument('--goals', type=int, default=4, help='The number of goal locations.')
    parser.add_argument('--gridwidth', type=int, default=20, help='The width of the grid world.')
    parser.add_argument('--gridheight', type=int, default=20, help='The height of the grid world.')
    parser.add_argument('--rstdev', type=float, default=0.1, help='The (known) standard deviation of the reward function.')
    parser.add_argument('--gstdev', type=int, default=1, help='The (known) standard deviation of the goal locations.')
    parser.add_argument('--maxmoves', type=int, default=100, help='The maximum number of moves per episode.')
    # Q-Learning arguments
    parser.add_argument('--epsilon', type=float, default=0.1, help='The exploration rate for the Q-Learning agent. range: [0,1]')
    parser.add_argument('--alpha', type=float, default=0.1, help='The learning rate for the Q-Learning agent. range: [0,1]')
    parser.add_argument('--gamma', type=float, default=0.1, help='The discount factor for the Q-Learning agent. range: [0,1]')
    args = parser.parse_args()

    agents = get_agents(args)
    classes = [MdpClass(i, args) for i in range(args.classes)]
    domains = [create_domain(d, args, classes) for d in range(args.domains)]
    
    agent_rewards = []
    for agent in agents:
        print 'Agent: {0}'.format(agent.name)
        first_episode_rewards = np.zeros((args.domains))
        for domain_idx, domain in enumerate(domains):
            agent.domains[domain.task_id] = domain
            domain.agent = agent
            steps = 0
            reward = domain.play_episode()
            first_episode_rewards[domain_idx] = reward
        agent_rewards.append(first_episode_rewards)
        
    agent_colors = ['red','blue','yellow', 'green', 'orange', 'purple', 'brown'] # max 7 agents
    ax = plt.subplot(111)
    plt.xlim((0,args.domains))
    xvals = np.arange(args.domains)
    for i,rewards in enumerate(agent_rewards):
        # Plot each series
        plt.plot(xvals, rewards, label=agents[i].name, color=agent_colors[i])
    plt.xlabel('# Experienced Environments')
    plt.ylabel('Total Reward Episode 1')
    #plt.ylim([0,1])
    plt.title('{0}x{1} Map, {2} Locations, {3} Colors'.format(args.gridwidth, args.gridheight, args.goals, args.colors))
    # Shink current axis by 25%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('{0}.pdf'.format('test_locations'))
    plt.clf()


