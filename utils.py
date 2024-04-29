import gymnasium as gym
import h5py
import numpy as np

from tianshou.data import ReplayBuffer
from tianshou.utils import RunningMeanStd



import pickle
from tianshou.data import ReplayBuffer

def load_buffer(buffer_path: str, size=10000) -> ReplayBuffer:
    with open(buffer_path, 'rb') as f:
        custom_buffer = pickle.load(f)

    obs_list = custom_buffer.observations[:size]
    act_list = custom_buffer.actions[:size]
    reward_list = custom_buffer.rewards[:size]
    next_obs_list = custom_buffer.next_observations[:size]
    done_list = custom_buffer.dones[:size]
    timeouts_list = custom_buffer.timeouts[:size]

    return ReplayBuffer.from_data(
        obs=obs_list.squeeze(),
        act=act_list.squeeze(),
        rew=reward_list.squeeze(),
        done=(done_list != 0).astype(bool).squeeze(),
        obs_next=next_obs_list.squeeze(),
        terminated=(done_list != 0).astype(bool).squeeze(),
        truncated=(timeouts_list != 0).astype(bool).squeeze(),
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