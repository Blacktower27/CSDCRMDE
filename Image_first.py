import numpy as np
import torch
from itertools import count

from env import Env
import config
import math
import csv

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env = Env()
env.seed(config.RANDOM_SEED)
torch.manual_seed(0)
num_state = env.n_observations
num_action = env.node_num


class Image_first():
    def __init__(self):
        super(Image_first, self).__init__()

    def select_action(self, env, task, state):
        i =0
        delay = math.inf
        for c in range(0,num_action): #find the node that has the smallest image download time
            if task.mem > env.node[c].mem or env.node[c].cpu_freq <= 0:
                continue
            else:
                image_download_time = env.imgae_download(task, c)
                if image_download_time < delay:
                    i = c
                    delay = image_download_time
        return i, 0

def main():
    agent = Image_first()

    return_list = []
    total_times = []
    total_energys = []
    record = []
    total_download_time=[]

    save_interval = 10  # Set the save interval to save every 10 epochs
    save_filename = 'log/Image_first' + str(config.e1) + str(config.e2) + 'node' + str(config.EDGE_NODE_NUM) + 'task'+str(config.max_tasks)+'_'+str(config.min_tasks)+ '.csv' 
    for i_epoch in range(config.epoch):
        ep_reward = []
        ep_action_prob = 0
        env.reset()

        cnt = 0
        for t in count():
            done, _, idx = env.env_up()

            if done:
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
            t_on_n = [0] * env.node_num# Record the number of tasks assigned to each node
            tasks = []
            while env.task and env.task[0].start_time == env.time:
                temp += 1
                curr_task = env.task.pop(0)
                state = env.get_obs(curr_task)
                states.append(state)
                tasks.append(curr_task)
                action, action_prob = agent.select_action(env, curr_task, state)# Select an action for the current task
                actions.append(action)
                action_probs.append(action_prob)
            for n_id in actions:
                t_on_n[n_id] += 1
            next_states, rewards, done, download_finish_time = env.step(tasks, actions, t_on_n)# Get the reward next state and other information of all tasks in the current time slot
            for i in range(0, temp):
                cnt += 1
                reward = rewards[i]
                ep_reward.append(reward)
                # Check if the save interval is reached, and if so save the data
                if i_epoch % save_interval == 0 and i_epoch > 0:
                    with open(save_filename, mode='w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            ["Episode", "Reward", "Total Time", "Total Energy", "complete ratio", "total_download_time",
                             "Fail Times"])
                        writer.writerows(record)

if __name__ == '__main__':
    main()