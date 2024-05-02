import numpy as np
from tianshou.utils import RunningMeanStd
import torch
import pickle
from tianshou.data import ReplayBuffer

def load_buffer_from_pkl(buffer_path: str, size=80000) -> ReplayBuffer:
    with open(buffer_path, 'rb') as f:
        custom_buffer = pickle.load(f)

    obs_list = custom_buffer.observations[:size]
    act_list = custom_buffer.actions[:size]
    reward_list = custom_buffer.rewards[:size]
    reward_list[reward_list == -100] = -1
    reward_list[reward_list == 10] = 1
    next_obs_list = custom_buffer.next_observations[:size]
    done_list = custom_buffer.dones[:size]

    return ReplayBuffer.from_data(
        obs=obs_list.squeeze(),
        act=act_list.squeeze(),
        rew=reward_list.squeeze(),
        done=(done_list != 0).astype(bool).squeeze(),
        obs_next=next_obs_list.squeeze(),
        terminated=(done_list != 0).astype(bool).squeeze(),
        truncated= np.zeros(size, dtype=bool)
    )



def load_buffer_from_csv(buffer_path: str, size = 200000) -> ReplayBuffer:
    def process_string(s):
        s = s.replace('\n', '')
        return np.array([float(num) for num in s.strip('[]').split()])
    import pandas as pd
    data = pd.read_csv(buffer_path)
    action_list = np.array(data["action"])
    obs_list = np.array(data["obs"])
    obs_list = np.array([process_string(s) for s in obs_list])
    reward_list = np.array(data["reward"])
    next_obs_list = np.array(data["next_obs"])
    next_obs_list = np.array([process_string(s) for s in next_obs_list])
    done_list = np.array(data["done"])
    return ReplayBuffer.from_data(
        obs=obs_list[:size],
        act=action_list[:size],
        rew=reward_list[:size],
        done=done_list[:size],
        obs_next=next_obs_list[:size],
        terminated=done_list[:size],
        truncated= np.zeros(size, dtype=bool)
    )


def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer,
) -> tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs - obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next - obs_rms.mean) / np.sqrt(
        obs_rms.var + _eps,
    )
    return replay_buffer, obs_rms


def save(args, save_name, model, ep=None):
    import os
    save_dir = './IQL_trained_models/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not ep == None:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + str(ep) + ".pth")
    else:
        torch.save(model.state_dict(), save_dir + args.run_name + save_name + ".pth")

def collect_random(env, dataset, num_samples=200):
    state, _= env.reset()
    for _ in range(num_samples):
        action = env.action_space.sample()
        next_state, reward, done, tr, _ = env.step(action)
        dataset.add(state, action, reward, next_state, done)
        state = next_state
        if done:
            state,_ = env.reset()