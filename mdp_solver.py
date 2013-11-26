import numpy as np
from gridworld import *

def value_iteration(width, height, goal, cell_rewards, discount=1.0, convergence=0.01):
	"""
	An implementation of value iteration to solve a gridworld MDP.
	"""
	# Create an arbitrary set of starting values (optimistic initialization)
	cell_values = np.zeros((width, height)) - 1000000

	# Initialize delta
	delta = 10000

	while delta > convergence:
		delta = 0
		# Check every state
		for x in range(width):
			for y in range(height):
				v = cell_values[x,y]

				if (x,y) == goal:
					cell_values[x,y] = 0
				else:
					# v' is the max possible reward of the next state plus the value of that next state
					if x > 0:
						cell_values[x,y] = max(cell_values[x,y], cell_rewards[x-1,y] + discount*cell_values[x-1,y])
					if x < (width - 1):
						cell_values[x,y] = max(cell_values[x,y], cell_rewards[x+1,y] + discount*cell_values[x+1,y])
					if y > 0:
						cell_values[x,y] = max(cell_values[x,y], cell_rewards[x,y-1] + discount*cell_values[x,y-1])
					if y < (height - 1):
						cell_values[x,y] = max(cell_values[x,y], cell_rewards[x,y+1] + discount*cell_values[x,y+1])
				
				# Calculate the change in our estimate of V(s)
				cur_delta = abs(v - cell_values[x,y])

				# Update the delta if this is bigger than the largest we've seen thus far
				if cur_delta > delta:
					delta = cur_delta
	return cell_values

def value_iteration_to_policy(width, height, goal, cell_rewards, discount=1.0, convergence=0.01):
	cell_values = value_iteration(width, height, goal, cell_rewards, discount=discount, convergence=convergence)
	policy = np.zeros((width, height))
	for x in range(width):
		for y in range(width):
			max_action = None
			max_value = None
			if x > 0:
				max_action = LEFT
				max_value = cell_values[x-1,y]
			if x < (width - 1):
				v = cell_values[x+1,y]
				if max_value is None or max_value < v:
					max_action = RIGHT
					max_value = v
			if y > 0:
				v = cell_values[x,y-1]
				if max_value is None or max_value < v:
					max_action = UP
					max_value = v
			if y < (height - 1):
				v = cell_values[x,y+1]
				if max_value is None or max_value < v:
					max_action = DOWN
					max_value = v
			if max_action is None:
				raise Exception('Invalid cell')
			policy[x,y] = max_action
	return policy

if __name__ == "__main__":
    agent = Agent(None)
    color_means = (-4,-5,-2,-3)
    color_scores = np.array([random.normalvariate(mu, 1) for mu in color_means])
    world = GridWorld(0, agent, 10, 10, 100, color_scores, (0,0), None)
    world.start()
    world.print_world()
    values = value_iteration(world.width, world.height, world.goal, world.cell_values)
    world.print_world(values)