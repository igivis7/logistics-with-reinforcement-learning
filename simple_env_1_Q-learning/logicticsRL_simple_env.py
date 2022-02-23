# ship size is 10
# 5 types of boxes with sizes: 1,2,3,4,5
# 1 box each
# reward for a box is equal to box size (for simplicity)
# number of unused ship cells is negative reward (-1, -2 or -5 ...)
# observation space is number of filled ship cells and number of available items in depot
# state is an element of observation space [used_ship_cells, n_of_items_0, n_of_items_1, n_of_items_2, ...]
# action = loading a single box
# done when empty cells = 0 or N_of empty cells < min(box)
# solutions:
#       10: {5,4,1}{2,3,5}{1,2,3,4}
#        9: {2,3,4}
#        8: {1,3,4}{3,5}{2,5,1}
#        7: {3,4}{2,5}{4,2,1}


import gym
import numpy as np
import random


class LogisticsRLBaseEnv(gym.Env):
    # === INIT ====
    def __init__(self):
        # 5 types of items, 1 or more - how many items are not yet used/processed
        # self.initial_items_depot = {'item_0': 10, 'item_1': 10, 'item_2': 10, 'item_3': 10, 'item_4': 10}
        # implementation of random number of items in the depot
        min_n_of_items = 1
        max_n_of_items = 1
        self.initial_items_depot = [random.randint(min_n_of_items, max_n_of_items),
                                    random.randint(min_n_of_items, max_n_of_items),
                                    random.randint(min_n_of_items, max_n_of_items),
                                    random.randint(min_n_of_items, max_n_of_items)]
        # item weight is basically reward and state change by loading it
        self.items_weight = [1, 2, 3, 4]
        # state to remember - position in observation space: used ship cells, left items of each type
        self.current_state = [0] + self.initial_items_depot
        # the number of free cells on the ship
        self.ship_size = 10
        self.free_ship_cells = 0
        # each action is loading a single box, max 5 boxes, all of them
        # actions ids are: 0,1,2,3,4
        self.action_space = gym.spaces.Discrete(len(self.initial_items_depot))
        # observation space is list of all states: 10 filled ship cells + 0 = empty ship. 11 in total
        # and
        # each depot slot status
        # => the dimensions of the observation space are 1 + len(initial_items_depot)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(1 + self.initial_items_depot.__len__(), dtype=np.int16),
            high=np.array([self.ship_size] + self.initial_items_depot),
            dtype=np.int16)

    # === STEP ====
    def step(self, action_id):
        reward_internal = None  # to ensure that variable always assigned
        done_internal = False
        self.free_ship_cells = self.ship_size - self.current_state[0]

        # to check validity of an action
        # try:
        #     self.current_state[action_id + 1]
        # except:
        #     raise Exception("Action is not valid")
        if action_id not in np.arange(0, len(self.initial_items_depot)):
            raise Exception("Action is not valid")

        if (self.current_state[action_id + 1] > 0) and \
                (self.free_ship_cells >= self.items_weight[action_id]):
            self.current_state[0] += self.items_weight[action_id]
            self.current_state[action_id + 1] -= 1
            reward_internal = self.items_weight[action_id]
        else:
            reward_internal = 0 

        # the process is done when there are no items in initial_items_depot
        # or
        # when the smallest item in initial_items_depot is bigger that the left space on the ship
        if (sum(self.current_state[1:]) <= 0) \
                or \
                (min(
                    [self.items_weight[ind] for ind, val in enumerate(self.current_state[1:]) if val > 0]
                ) > self.free_ship_cells):
            done_internal = True

        info_internal = {}

        return self.current_state, reward_internal, done_internal, info_internal

    # === RESET ====
    def reset(self):
        # self.current_state = 0
        self.current_state = [0] + self.initial_items_depot
        return self.current_state

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
state_space_size = env.observation_space.high + 1  # +1 because we need 0-state as well

# adding 1 to state_space_size in the q_table initialization is needed to
# include empty ship state and 0-item for each item_type state
q_table = np.zeros(np.append(state_space_size, action_space_size))
print(state_space_size)
print(q_table[:, 0, 0, 0, 0, :])

# =============================================================================

# some implementation of Q-learning


num_episodes = 1000  # default 1000
max_steps_per_episode = 40  # default 10

learning_rate = 0.01
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01

exploration_decay_rate = 0.01  # if we decrease it, will learn slower # default 0.01

rewards_all_episodes = []

# Q-Learning algorithm
for episode in range(num_episodes):
    state = env.reset()

    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        s_ship, s_item_0, s_item_1, s_item_2, s_item_3 = state

        # Exploration -exploitation trade-off
        exploration_rate_threshold = random.uniform(0, 1)
        if exploration_rate_threshold > exploration_rate:
            action = np.argmax(q_table[s_ship,
                               s_item_0,
                               s_item_1,
                               s_item_2,
                               s_item_3,
                               :])
        else:
            action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)
        new_s_ship, new_s_item_0, new_s_item_1, new_s_item_2, new_s_item_3 = new_state

        # Update Q-table for Q(s,a)
        q_table[s_ship,
                s_item_0,
                s_item_1,
                s_item_2,
                s_item_3,
                action] = (1 - learning_rate) * q_table[s_ship,
                                                        s_item_0,
                                                        s_item_1,
                                                        s_item_2,
                                                        s_item_3,
                                                        action] + \
            learning_rate * (reward + discount_rate * np.max(q_table[new_s_ship,
                                                             new_s_item_0,
                                                             new_s_item_1,
                                                             new_s_item_2,
                                                             new_s_item_3,
                                                             :]))

        state = new_state
        rewards_current_episode += reward

        if done:
            break

    # Exploration rate decay
    exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * \
        np.exp(-exploration_decay_rate * episode)

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
# print(q_table[:, 0, 0, 0, 0, :].round(2))
print(q_table[:, state_space_size[1]-1,
      state_space_size[2]-1,
      state_space_size[3]-1,
      state_space_size[4]-1, :].round(2))

pass  # just for a breakpoint
# _  # just for a breakpoint