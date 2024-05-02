import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
from iql_buffer import ReplayBuffer
import glob
from utils import save, collect_random
from agent import IQL
from torch.utils.data import DataLoader, TensorDataset
import pickle
import cellworld_gym as cwg
def get_config():
    parser = argparse.ArgumentParser(description='RL_IQL')
    parser.add_argument("--run_name", type=str, default="IQL-discrete", help="Run name")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=200, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=2, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--eval_every", type=int, default=2, help="Evaluate every x episodes, default: 10")
    args = parser.parse_args()
    return args

def prep_dataloader(batch_size=256, size=100_000):
    tensors = {}
    with open("first_try.pkl", 'rb') as f:
        custom_buffer = pickle.load(f)
    # make sure the data are in the right format
    obs_list = custom_buffer.observations[:size].squeeze(axis=1)
    act_list = custom_buffer.actions[:size].squeeze(axis=1)
    reward_list = custom_buffer.rewards[:size].squeeze()
    reward_list[reward_list == -100] = -1
    reward_list[reward_list == 10] = 1
    next_obs_list = custom_buffer.next_observations[:size].squeeze(axis=1)
    done_list = custom_buffer.dones[:size].squeeze(axis=1)
    # print(obs_list.shape)
    # print(act_list.shape)
    # print(reward_list.shape)
    # print(next_obs_list.shape)
    # print(done_list.shape)
    # (100000, 11)
    # (100000, 1)
    # (100000,)
    # (100000, 11)
    # (100000,)
    tensors["observations"] = torch.from_numpy(obs_list).float()
    tensors["actions"] = torch.from_numpy(act_list).float()
    tensors["rewards"] = torch.from_numpy(reward_list).float()
    tensors["next_observations"] = torch.from_numpy(next_obs_list).float()
    tensors["dones"] = torch.from_numpy(done_list).long()

    tensordata = TensorDataset(tensors["observations"],
                               tensors["actions"],
                               tensors["rewards"],
                               tensors["next_observations"],
                               tensors["dones"])
    dataloader = DataLoader(tensordata, batch_size=batch_size, shuffle=True)
    return dataloader

def evaluate(env, policy, eval_runs=5):
    reward_batch = []
    for i in range(eval_runs):
        state,_ = env.reset()
        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)
            state, reward, done, tr ,_ = env.step(action)
            rewards += reward
            if (done or tr):
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

def train(config):
    env = gym.make("CellworldBotEvade-v0",
             world_name="21_05",
             use_lppos=False,
             use_predator=True,
             max_step=300,
             time_step=0.25,
             render=False,
             real_time=False,
             reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = prep_dataloader(batch_size=config.batch_size)

    batches = 0
    average10 = deque(maxlen=10)

    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                device=device)
    for i in range(1, config.episodes+1):
        for batch_idx, (states, actions, rewards, next_states, dones) in enumerate(dataloader):
            states = states.to(device)
            actions = actions.to(device)
            rewards = rewards.to(device)
            next_states = next_states.to(device)
            dones = dones.to(device)
            policy_loss, critic1_loss, critic2_loss, value_loss = agent.learn((states, actions, rewards, next_states, dones))
            batches += 1
        if i % config.eval_every == 0:
            eval_reward = evaluate(env, agent)
            average10.append(eval_reward)
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches))
        if i % config.save_every == 0:
            save(config, save_name="IQL", model=agent.actor_local)


if __name__ == "__main__":
    config = get_config()
    train(config)