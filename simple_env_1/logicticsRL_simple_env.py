# ship size is 10
# 5 types of boxes with sizes: 1,2,3,4,5
# 1 box each
# reward for a box is equal to box size (for simplicity)
# number of unused ship cells is negative reward (-1, -2 or -5 ...)
# observation space is number of filled ship cells
# action = loading a single box
# done when empty cells = 0 or N_of empty cells < min(box)
# solutions:
#       10: {5,4,1}{2,3,5}{1,2,3,4}
#        9: {2,3,4}
#        8: {1,3,4}{3,5}{2,5,1}
#        7: {3,4}{2,5}{4,2,1}


import gym
import numpy as np


class LogisticsRLBaseEnv(gym.Env):
    # === INIT ====
    def __init__(self):
        # 5 types of items, 1 or more - how many items are not yet used/processed
        self.items_depot = {'item_0': 1, 'item_1': 1, 'item_2': 1, 'item_3': 1, 'item_4': 1}
        # item weight is basically reward and state change by loading it
        self.items_weight = {'item_0': 1, 'item_1': 2, 'item_2': 3, 'item_3': 4, 'item_4': 5}
        # state to remember - position in observation space
        self.current_state = 0
        # the number of free cells on the ship
        self.ship_size = 10
        # each action is loading a single box, max 5 boxes, all of them
        # actions ids are: 0,1,2,3,4
        self.action_space = gym.spaces.Discrete(5)
        # observation space is list of all states: 10 filled ship cells + 0 = empty ship
        # 11 in total
        self.observation_space = gym.spaces.Discrete(11)

    # === STEP ====
    def step(self, action_id):
        reward_internal = 0  # to ensure that variable always assigned
        done_internal = False
        free_ship_cells = self.ship_size - self.current_state

        if (action_id == 0) and \
                (self.items_depot['item_0'] > 0) and \
                (free_ship_cells >= self.items_weight['item_0']):
            self.current_state += self.items_weight['item_0']
            reward_internal = self.items_weight['item_0']
            self.items_depot['item_0'] -= 1
        elif (action_id == 1) and \
                (self.items_depot['item_1'] > 0) and \
                (free_ship_cells >= self.items_weight['item_1']):
            self.current_state += self.items_weight['item_1']
            reward_internal = self.items_weight['item_1']
            self.items_depot['item_1'] -= 1
        elif (action_id == 2) and \
                (self.items_depot['item_2'] > 0) and \
                (free_ship_cells >= self.items_weight['item_2']):
            self.current_state += self.items_weight['item_2']
            reward_internal = self.items_weight['item_2']
            self.items_depot['item_2'] -= 1
        elif (action_id == 3) and \
                (self.items_depot['item_3'] > 0) and \
                (free_ship_cells >= self.items_weight['item_3']):
            self.current_state += self.items_weight['item_3']
            reward_internal = self.items_weight['item_3']
            self.items_depot['item_3'] -= 1
        elif (action_id == 4) and \
                (self.items_depot['item_4'] > 0) and \
                (free_ship_cells >= self.items_weight['item_4']):
            self.current_state += self.items_weight['item_4']
            reward_internal = self.items_weight['item_4']
            self.items_depot['item_4'] -= 1

        # the process is done when there are no items in items_depot
        # or
        # when the smallest item in items_depot is bigger that the left space on the ship
        if (sum(self.items_depot.values()) <= 0) \
                or \
                (min(
                    [self.items_weight[key] for key, value in self.items_depot.items() if value > 0]
                ) > free_ship_cells):
            done_internal = True

        info_internal = {}

        return self.current_state, reward_internal, done_internal, info_internal

    # === RESET ====
    def reset(self):
        state_internal = 0
        self.current_state = 0  # just to ensure that it is everywhere reset
        return state_internal

    # === RENDER ====
    # not used in current implementation
    def render(self, **kwargs):
        pass

    # === CLOSE ====
    # not used in current implementation
    def close(self):
        pass


# =============================================================================


env = LogisticsRLBaseEnv()

action_space_size = env.action_space.n
state_space_size = env.observation_space.n

q_table = np.zeros((state_space_size, action_space_size))
print(q_table)

# =============================================================================

# some implementation of Q-learning

import random

num_episodes = 1000
max_steps_per_episode = 100  # but it won't go higher than 1

learning_rate = 0.01
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate = 0.01  # if we decrease it, will learn slower

rewards_all_episodes = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):

        # Exploration -exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Update Q-table for Q(s,a)
        q_table[state, action] = (1 - learning_rate) * q_table[state, action] + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate * episode)

    rewards_all_episodes.append(rewards_current_episode)

# Calculate and print the average reward per 10 episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes / 100)
count = 100
print("********** Average  reward per thousand episodes **********\n")

for r in rewards_per_thousand_episodes:
    print(count, ": ", str(sum(r / 100)))
    count += 100

# Print updated Q-table
print("\n\n********** Q-table **********\n")
print(q_table)