import numpy as np
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
from env import Env  # Importing the environment module
import config
import csv
import math

# Check if GPU is available, else use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Create an instance of the environment
env = Env()
env.seed(config.RANDOM_SEED)  # Set a random seed for reproducibility
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

# Define a named tuple 'Transition' for storing experience replay transitions
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done'])
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])

# Initialize weights using a custom function
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

# Definition of the Q-network class
class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        # Define layers for the Q-network
        self.gru1 = nn.GRUCell(num_state, 128)
        self.gru2 = nn.GRUCell(128, 64)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 16)
        self.action_head = nn.Linear(16, num_action)

    def forward(self, x):
        # Define forward pass for the Q-network
        x = F.leaky_relu(self.gru1(x))
        x = F.leaky_relu(self.gru2(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.action_head(x)
        return x

# Definition of the DQN (Deep Q-Network) class
class DQN():
    # Hyperparameters for DQN
    clip_param = 0.2
    max_grad_norm = 0.5
    buffer_capacity = 20000
    minimal_size = 2000
    batch_size = 128
    epsilon = 0.01

    def __init__(self):
        super(DQN, self).__init__()
        # Initialize DQN variables and networks
        self.buffer = []  # Experience replay buffer
        self.counter = 0

        self.q_net = Qnet().to(device)  # Q-network
        self.target_q_net = Qnet().to(device)  # Target Q-network
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)  # Using Adam optimizer
        self.gamma = 0.98  # Discount factor
        self.target_update = 5  # Target network update frequency
        self.count = 0  # Counter, records the number of updates
        self.tau = 0.005

    def select_action(self, env, task, state):
        # Use epsilon-greedy strategy to select actions
        if np.random.random() < self.epsilon:
            # Exploration: Randomly choose an action with epsilon probability
            random_numbers = [random.random() for _ in range(num_action)]
            count = 0
            for i in range(env.node_num):
                # Filter out actions that are not feasible
                flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
                if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                    count += 1
                    random_numbers[i] = -math.inf
                    if count == num_action:
                        print("DQN can't find fit action")
                        print('image:', flag)
                        print('memory:', task.mem > env.node[i].mem)
                        print('cpu:', env.node[i].cpu_freq <= 0)
                        return -1, 0
            action = np.array(random_numbers).argmax().item()
        else:
            # Exploitation: Choose the action with the highest Q-value
            state = torch.tensor([state], dtype=torch.float).to(device)
            with torch.no_grad():
                action_q = self.q_net(state)
            count = 0
            for i in range(env.node_num):
                # Filter out actions that are not feasible
                flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
                if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                    count += 1
                    action_q[0, i] = -math.inf
                    if count == num_action:
                        print("DQN can't find fit action")
                        print('image:', flag)
                        print('memory:', task.mem > env.node[i].mem)
                        print('cpu:', env.node[i].cpu_freq <= 0)
                        return -1, 0
            action = action_q.argmax().item()
        return action, 0

    def soft_update(self, net, target_net):
        # Soft update for target network
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) + param.data * self.tau)

    def store_transition(self, transition):
        # Store experience replay transitions in the buffer
        if len(self.buffer) < self.buffer_capacity:
            self.buffer.append(transition)
        else:
            self.buffer.pop(0)
            self.buffer.append(transition)
        return len(self.buffer) % self.minimal_size == 0

    def update(self):
        # Perform DQN network update
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        actions = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(device)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)
        next_states = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float).view(-1, 1).to(device)

        q_values = self.q_net(states).gather(1, actions)
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()
        self.soft_update(self.q_net, self.target_q_net)

# Main function
def main():
    agent = DQN()
    total_times = []
    total_energys = []
    total_download_time = []
    record = []
    fail_time = 0
    i_epoch = 0
    save_interval = 10  # Set the saving interval, save every 10 epochs
    save_filename = '../log/fcDQN' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'cpu' + str(
        config.node_cpu_freq_max) + 'task' + str(config.max_tasks) + '_' + str(
        config.min_tasks) + '.csv'  # Set the filename for saving
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()

        cnt = 0
        fail = False
        for t in count():
            done, _, idx = env.env_up()

            if done:
                # End of an episode
                i_epoch += 1
                total_times.append(env.total_time)
                total_energys.append(env.total_energy)
                total_download_time.append(env.download_time)
                num_on_time = env.num_on_time
                total_task = env.total_task
                complete_ratio = num_on_time / total_task
                print(
                    'Episode: {}, reward: {}, total_time: {}, total_energy: {}, complet_ratio: {}, donwload time: {}'.format(
                        i_epoch, round(np.mean(ep_reward), 3), env.total_time, env.total_energy, complete_ratio,
                        env.download_time))

                record.append([i_epoch, round(np.mean(ep_reward), 3), env.total_time,
                               env.total_energy, complete_ratio, env.download_time])
                break
            temp = 0
            actions = []
            states = []
            action_probs = []
            t_on_n = [0] * env.node_num
            tasks = []
            while env.task and env.task[0].start_time == env.time:
                temp += 1
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                states.append(state)
                tasks.append(curr_task)
                action, action_prob = agent.select_action(env, curr_task, state)
                if action == -1:
                    fail = True
                    break
                actions.append(action)
                action_probs.append(action_prob)
            if fail == True:
                # DQN couldn't find a fit action
                fail_time += 1
                i_epoch -= 1
                break
            for n_id in actions:
                t_on_n[n_id] += 1
            next_states, rewards, done, download_finish_time = env.step(tasks, actions, t_on_n)

            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                trans = Transition(states[i], actions[i], reward, action_probs[i], next_states[i], done)
                agent.store_transition(trans)
                if cnt > 3000:
                    if cnt % 1000 == 0:
                        agent.update()

        if i_epoch % save_interval == 0 and i_epoch > 0:
            # Save training progress to a CSV file
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
                     "Fail Times"])
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        # Save final training progress to a CSV file
        writer = csv.writer(file)
        writer.writerow(
            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
             "Fail Times"])
        record[0].append(fail_time)
        writer.writerows(record)

if __name__ == '__main__':
    main()
