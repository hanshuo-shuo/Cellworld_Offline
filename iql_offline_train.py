import gymnasium as gym
import numpy as np
from collections import deque
import torch
import argparse
from cellworld import World, World_statistics
from utils import save, collect_random
from agent import IQL
from torch.utils.data import DataLoader, TensorDataset
import pickle
import cellworld_gym as cwg
from video import save_video_output

def get_config():
    parser = argparse.ArgumentParser(description='RL_IQL')
    parser.add_argument("--run_name", type=str, default="IQL-discrete", help="Run name")
    parser.add_argument("--env", type=str, default="CartPole-v0", help="Gym environment name, default: Pendulum-v0")
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--eval_every", type=int, default=10, help="Evaluate every x episodes, default: 10")
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

def get_importance_states()->list:
    world = World.get_from_parameters_names("hexagonal", "canonical", "21_05")
    world_statistics = World_statistics.get_from_parameters_names("hexagonal", "21_05")
    m = max(world_statistics.visual_centrality_derivative)
    v = [1 if p > m * .1 else 0 for p in world_statistics.visual_centrality_derivative]
    world_statistics = World_statistics.get_from_parameters_names("hexagonal", "21_05")
    m = max(world_statistics.visual_centrality_derivative)
    v = [1 if p > m * .1 else 0 for p in world_statistics.visual_centrality_derivative]
    # get the locations for high central derivative
    locations = list()
    for cell_id, p in enumerate(v):
        if p == 1:
            l = world.cells[cell_id].location
            locations.append((l.x, l.y))
    return locations


def prep_dataloader_hc(batch_size=256):
    tensors = {}
    with open("new_data.pkl", 'rb') as f:
    # with open("new_data_real.pkl", 'rb') as f:
        custom_buffer = pickle.load(f)
        print(custom_buffer['observations'].shape)
    obs_list = custom_buffer['observations']
    act_list = custom_buffer['actions']
    reward_list = custom_buffer['rewards']
    reward_list[reward_list == -100] = -1
    reward_list[reward_list == 10] = 1
    next_obs_list = custom_buffer['next_observations']
    done_list = custom_buffer['dones']
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

def evaluate(env, policy, eval_runs=10):
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
    rewards_list = []
    policy_losses_list = []
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
    # dataloader = prep_dataloader(batch_size=config.batch_size)
    dataloader = prep_dataloader_hc(batch_size=config.batch_size)
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
            rewards_list.append(eval_reward)
            policy_losses_list.append(policy_loss)
            print("Episode: {} | Reward: {} | Polciy Loss: {} | Batches: {}".format(i, eval_reward, policy_loss, batches))
        if i % config.save_every == 0:
            save(config, save_name="IQL_400", model=agent.actor_local)
        # plot and save plot
        import matplotlib.pyplot as plt
        plt.plot(rewards_list)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Reward over Episodes')
        plt.savefig('reward_plot.png')
        plt.close()
        plt.plot(policy_losses_list)
        plt.xlabel('Episode')
        plt.ylabel('Policy Loss')
        plt.title('Policy Loss over Episodes')
        plt.savefig('policy_loss_plot.png')
        plt.close()

def evaluate_after_train(eval_runs=100):
    env = gym.make("CellworldBotEvade-v0",
                world_name="21_05",
                use_lppos=False,
                use_predator=True,
                max_step=300,
                time_step=0.25,
                render=False,
                real_time=False,
                reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    reward_batch = []
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                device="cpu")
    agent.actor_local.eval()
    agent.actor_local.load_state_dict(torch.load("IQL_trained_models/IQL-discreteIQL_400.pth"))
    for i in range(eval_runs):
        state, _ = env.reset()
        rewards = 0
        while True:
            action = agent.get_action(state, eval=True)
            state, reward, done, tr, _ = env.step(action)
            rewards += reward
            if (done or tr):
                break
        reward_batch.append(rewards)
    print(np.mean(reward_batch))

def kl_divergence():
    from scipy.stats import entropy
    with open('first_try.pkl', 'rb') as file:
        replay_buffer = pickle.load(file)

    observations_for_action = replay_buffer.observations.squeeze(axis=1)[:100000]
    real_actions = replay_buffer.actions.squeeze(axis=1)[:100000]
    #get the actions from the model
    env = gym.make("CellworldBotEvade-v0",
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   max_step=300,
                   time_step=0.25,
                   render=False,
                   real_time=False,
                   reward_function=cwg.Reward({"puffed": -1, "finished": 1}))
    agent = IQL(state_size=env.observation_space.shape[0],
                action_size=env.action_space.n,
                device="cpu")
    agent.actor_local.eval()
    agent.actor_local.load_state_dict(torch.load("IQL_trained_models/IQL-discreteIQL_400.pth"))
    predicted_action = agent.get_action(observations_for_action, eval=True)
    ## get the probabilities of the actions
    actions = np.array(predicted_action)
    unique_actions, action_counts = np.unique(actions, return_counts=True)
    total_actions = len(actions)
    action_probabilities = action_counts / total_actions
    ## get the probabilities of the real actions
    real_actions = np.array(real_actions)
    unique_real_actions, real_action_counts = np.unique(real_actions, return_counts=True)
    total_real_actions = len(real_actions)
    real_action_probabilities = real_action_counts / total_real_actions
    # transform the probabilities to the exp-probabilities
    real_action_probabilities = np.exp(real_action_probabilities)/np.sum(np.exp(real_action_probabilities))
    action_probabilities = np.exp(action_probabilities)/np.sum(np.exp(action_probabilities))
    kl_div = entropy(real_action_probabilities, action_probabilities)
    print(kl_div)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.hist(predicted_action, bins=50, color='blue', alpha=0.7)
    plt.title('Trained Actions IQL')
    plt.xlabel('Action')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    config = get_config()
    train(config)
    # evaluate_after_train()
    # kl_divergence()