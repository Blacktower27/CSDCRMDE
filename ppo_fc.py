import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple
from itertools import count
from env import Env
import config
import csv


# Set the device to GPU if available, otherwise use CPU
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Initialize the environment
env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num

# Define a named tuple for storing the transition information
Transition = namedtuple(
    'Transition',
    ['state', 'action', 'reward', 'a_log_prob', 'next_state', 'done']
)

# Define a named tuple for recording training information
TrainRecord = namedtuple('TrainRecord', ['episode', 'reward'])

# Function to initialize the weights of the neural network
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0., std=0.1)
        nn.init.constant_(m.bias, 0.1)

# Actor neural network class
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.action_head = nn.Linear(32, num_action)

    def forward(self, x):
        # Forward pass through the actor network
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.action_head(x)
        action_prob = F.softmax(x, dim=1, dtype=torch.double)
        return action_prob

# Critic neural network class
class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc2 = nn.Linear(64, 16)
        self.state_value = nn.Linear(16, 1)

    def forward(self, x):
        # Forward pass through the critic network
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        value = self.state_value(x)
        return value

# Proximal Policy Optimization (PPO) class
class PPO():
    clip_param = 0.2
    max_grad_norm = 0.5
    ppo_epoch = 5
    buffer_capacity = 1000
    batch_size = 128

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor().to(device)
        self.critic_net = Critic().to(device)
        self.buffer = []
        self.counter = 0
        self.gamma = 0.98
        self.lmbda = 0.95

        # Set up optimizers for the actor and critic networks
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=1e-4)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), lr=3e-4)

    def select_action(self, env, task, state):
        # Given the current state, select an action using the actor network

        state = torch.tensor([state], dtype=torch.float).to(device)
        action_mask = torch.tensor([1] * env.node_num, dtype=torch.float).to(device)
        count = 0
        for i in range(env.node_num):
            # Apply action mask based on certain conditions
            flag = env.image[task.image_id].image_size > env.node[i].disk and env.node[i].image_list[task.image_id] == 0
            if flag or task.mem > env.node[i].mem or env.node[i].cpu_freq <= 0:
                count += 1
                action_mask[i] = 0
                if count == num_action:
                    print("ppo_fc can't find fit action")
                    print('image:', flag)
                    print('memory:', task.mem > env.node[i].mem)
                    print('cpu:', env.node[i].cpu_freq <= 0)
                    return -1, 0

        with torch.no_grad():
            action_prob = self.actor_net(state)
            action_prob = torch.mul(action_prob, action_mask)

        action_dist = torch.distributions.Categorical(action_prob)
        action = action_dist.sample().item()

        return action, 0

    def store_transition(self, transition):
        # Store a transition in the replay buffer
        self.buffer.append(transition)
        self.counter += 1
        return self.counter % self.buffer_capacity == 0

    def compute_advantage(self, gamma, lmbda, td_delta):
        # Compute the advantage for training using Generalized Advantage Estimation (GAE)

        td_delta = td_delta.detach().cpu().numpy()
        advantage_list = []
        advantage = 0.0
        for delta in td_delta[::-1]:
            advantage = gamma * lmbda * advantage + delta
            advantage_list.append(advantage)
        advantage_list.reverse()
        return torch.tensor(np.concatenate(advantage_list), dtype=torch.float)

    def update(self):
        # Update the actor and critic networks using Proximal Policy Optimization (PPO)
        states = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(device)
        next_states = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float).to(device)
        dones = torch.tensor([t.done for t in self.buffer], dtype=torch.float).to(device)
        rewards = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).to(device)
        actions = torch.tensor([t.action for t in self.buffer]).view(-1, 1).to(device)
        old_action_log_probs = torch.log(self.actor_net(states).gather(1, actions)).detach()

        for i in range(self.ppo_epoch):
            td_target = rewards + self.gamma * (self.critic_net(next_states).squeeze(1)) * (1 - dones)
            td_target = td_target.unsqueeze(1)
            td_delta = td_target - self.critic_net(states)
            advantage = self.compute_advantage(self.gamma, self.lmbda, td_delta).to(device)
            action_prob = torch.log(self.actor_net(states).gather(1, actions))
            ratio = torch.exp(action_prob - old_action_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage
            actor_loss = torch.mean(-torch.min(surr1, surr2))
            critic_loss = torch.mean(F.mse_loss(self.critic_net(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_net_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_net_optimizer.step()
        del self.buffer[:]


def main():
    # Main training loop

    agent = PPO()

    return_list = []
    total_times = []
    total_energys = []
    record = []
    save_interval = 10
    fail_time = 0
    save_filename = 'log/fcppo' + str(config.e1) + str(config.e2) + 'node' + str(
        config.EDGE_NODE_NUM) + 'cpu' + str(config.node_cpu_freq_max) + 'task' + str(
        config.max_tasks) + '_' + str(config.min_tasks) + '.csv'
    i_epoch = 0
    while i_epoch < config.epoch:
        ep_reward = []
        env.reset()
        fail = False

        cnt = 0
        for t in count():
            done, _, idx = env.env_up()

            if done:
                i_epoch += 1
                return_list.append(np.mean(ep_reward))
                total_times.append(env.total_time)
                total_energys.append(env.total_energy)
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
                if agent.store_transition(trans):
                    agent.update()

        if i_epoch % save_interval == 0 and i_epoch > 0:
            with open(save_filename, mode='w', newline='') as file:
                writer = csv.writer(file)

                # Write the CSV file header
                writer.writerow(
                    ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
                     "Fail Times"])
                # Write the data in memory to the CSV file
                writer.writerows(record)

    with open(save_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the CSV file header
        writer.writerow(
            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
             "Fail Times"])
        record[0].append(fail_time)

        # Write the data in memory to the CSV file
        writer.writerows(record)


if __name__ == '__main__':
    main()

