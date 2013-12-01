"""
A script to replicate experiment 1 from the Wilson et. al. (ICML'07) paper.
"""
import argparse
from gridworld import *
from qlearning import QAgent
from multitask import MultiTaskBayesianAgent, MdpClass, NormalInverseWishartDistribution
import random
import matplotlib.pyplot as plt
import numpy as np

def get_agents(args):
    agents = []
    for mdps in args.trainsize:
        if args.agent == 'qlearning':
            agents.append(QAgent(args.gridwidth, args.gridheight, args.colors, mdps+args.testsize, 'Q ({0} MPDs)'.format(mdps),\
                            args.epsilon, args.alpha, args.gamma))
        elif args.agent == 'multibayes':
            agents.append(MultiTaskBayesianAgent(args.gridwidth, args.gridheight, args.colors, mdps+args.testsize, args.rstdev, name='MTRL ({0} MDPs)'.format(mdps)))
        else:
            raise Exception('Unsupported agent type: ' + args.agent)
    return agents

def create_domain(task_id, args, classes):
    clazz = random.choice(classes)
    w = np.random.multivariate_normal(clazz.weights_mean, clazz.weights_cov)
    world = GridWorld(task_id, w, args.rstdev, None, args.gridwidth, args.gridheight, args.maxmoves, (0,0), None)
    return world

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tests an agent on a multi-task RL gridworld domain.')
    parser.add_argument('agent', choices=['qlearning', 'singlebayes', 'multibayes'])
    # General experiment arguments
    parser.add_argument('--classes', type=int, default=4, help='The number of classes that partition the set of MDPs.')
    parser.add_argument('--trainsize', type=int, nargs='*', default=[0,4,8,16], help='The number of MDP domains to train before evaluating.')
    parser.add_argument('--testsize', type=int, default=50, help='The number of test MDP domains to average over.')
    parser.add_argument('--teststeps', type=int, default=2500, help='The number of steps to measure in each test MDP.')
    parser.add_argument('--stepsize', type=int, default=10, help='The number of actions per measurement step in testing.')
    # Grid World arguments
    parser.add_argument('--colors', type=int, default=8, help='The number of colors in the MDP. Each cell is one color.')
    parser.add_argument('--gridwidth', type=int, default=15, help='The width of the grid world.')
    parser.add_argument('--gridheight', type=int, default=15, help='The height of the grid world.')
    parser.add_argument('--rstdev', type=float, default=0.1, help='The (known) standard deviation of the reward function.')
    parser.add_argument('--maxmoves', type=int, default=100, help='The maximum number of moves per episode.')
    # Q-Learning arguments
    parser.add_argument('--epsilon', type=float, default=0.1, help='The exploration rate for the Q-Learning agent. range: [0,1]')
    parser.add_argument('--alpha', type=float, default=0.1, help='The learning rate for the Q-Learning agent. range: [0,1]')
    parser.add_argument('--gamma', type=float, default=0.1, help='The discount factor for the Q-Learning agent. range: [0,1]')
    args = parser.parse_args()

    SIZE = args.colors * NUM_RELATIVE_CELLS

    agents = get_agents(args)
    niw_true = NormalInverseWishartDistribution(np.zeros(SIZE) - 3., 1., SIZE+2, np.identity(SIZE))
    true_params = [niw_true.sample() for _ in range(args.classes)]
    classes = [MdpClass(i, mean, cov) for i,(mean,cov) in enumerate(true_params)]
    test_domains = [create_domain(d, args, classes) for d in range(args.testsize)]
    train_domains = [create_domain(d, args, classes) for d in range(args.testsize, args.testsize+max(args.trainsize))]
    agent_rewards = []

    for agent,training in zip(agents,args.trainsize):
        print 'Agent: {0}'.format(agent.name)
        print 'Training...'
        for didx in range(training):
            domain = train_domains[didx]
            agent.domains[domain.task_id] = domain
            domain.agent = agent
            domain.play_episode()
        print 'Testing...'
        # TODO: Freeze the memory of the agent and restart it for every testing domain
        # so that it does not learn from previous test domains. Not a problem for Q-Learning.
        avg_rewards = np.array([0] * (args.teststeps / args.stepsize + 1))
        domain_weight = 1. / float(args.testsize)
        for domain in test_domains:
            agent.recent_rewards = [] # clear the reward history
            agent.domains[domain.task_id] = domain
            domain.agent = agent
            steps = 0
            domain.start()
            while steps < args.teststeps:
                # Keep restarting the episodes until we've gone the number of steps
                if not domain.episode_running:
                    domain.start()
                # Step forward in the domain
                domain.step()
                steps += 1
                # If we've taken a step's worth of actions, measure the cumulative rewards
                if steps % args.stepsize == 0:
                    step_idx = steps / args.stepsize
                    avg_rewards[step_idx] += domain_weight * sum(agent.recent_rewards)
            # Track the leftover steps in case stepsize is not a perfect divisor of teststeps.
            avg_rewards[-1] += domain_weight * sum(agent.recent_rewards)
        agent_rewards.append(avg_rewards)
        
    agent_colors = ['red','blue','yellow', 'green', 'orange', 'purple', 'brown'] # max 7 agents
    ax = plt.subplot(111)
    num_steps = args.teststeps / args.stepsize
    plt.xlim((0,num_steps))
    xvals = np.arange(num_steps)
    for i,rewards in enumerate(agent_rewards):
        # Plot each series
        plt.plot(xvals, rewards[0:num_steps], label=agents[i].name, color=agent_colors[i])
        #plt.fill_between(xvals, avg[i] + stderr[i], avg[i] - stderr[i], facecolor=colors[i], alpha=0.2)
    plt.xlabel('Number of Steps x {0}'.format(args.stepsize))
    plt.ylabel('Cumulative Reward')
    #plt.ylim([0,1])
    plt.title('{0}x{1} Map, Fixed Goal Location'.format(args.gridwidth, args.gridheight))
    # Shink current axis by 25%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig('{0}.pdf'.format('test'))
    plt.clf()


