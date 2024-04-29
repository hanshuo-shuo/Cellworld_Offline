import argparse
import datetime
import os
import pprint

import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
import torch
from vec_network import QRDQN
from utils import load_buffer
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DiscreteCQLPolicy
from tianshou.policy.base import BasePolicy
from tianshou.trainer import OfflineTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.space_info import SpaceInfo

import cellworld_gym as cwg


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        default="CellworldBotEvade-v0",
        help="The name of the OpenAI Gym environment to train on.",
    )
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--n-step", type=int, default=3)
    parser.add_argument("--target-update-freq", type=int, default=500)
    parser.add_argument("--min-q-weight", type=float, default=1.0)
    parser.add_argument("--num-quantiles",
                        type=int,
                        default=32,
                        help="The number of quantiles.")
    parser.add_argument("--expert-data",
                        type=str,
                        default="first_try.pkl",
                        help="The path to the expert data.")
    parser.add_argument(
        "--epoch",
        type=int,
        default=30,
        help="The number of epochs to train for.",
    )
    parser.add_argument(
        "--update-per-epoch",
        type=int,
        default=1000,
    )
    parser.add_argument("--test-num", type=int, default=5)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="The batch size for training.",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="The discount factor")
    parser.add_argument(
        "--logdir",
        type=str,
        default="log",
        help="The directory to save logs to.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="The device to train on (cpu or cuda).",
    )
    parser.add_argument(
        "--resume-path",
        type=str,
        default=None,
        help="The path to the checkpoint to resume from.",
    )
    parser.add_argument(
        "--resume-id",
        type=str,
        default=None,
        help="The ID of the checkpoint to resume from.",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard"
    )
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def test_cql() -> None:
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    args = get_args()
    env = gym.make(args.task,
                   world_name="21_05",
                   use_lppos=False,
                   use_predator=True,
                   max_step = 30,
                   time_step = 0.25,
                   reward_function=cwg.Reward({"puffed": -100, "finished": 10}))
    assert isinstance(env.action_space, gym.spaces.Discrete), "only discrete action space is supported"
    space_info = SpaceInfo.from_env(env)
    args.state_shape = space_info.observation_info.obs_shape
    args.action_shape = space_info.action_info.action_shape
    args.max_action = space_info.action_info.max_action
    args.min_action = space_info.action_info.min_action
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", args.min_action, args.max_action)

    args.state_dim = space_info.observation_info.obs_dim
    args.action_dim = space_info.action_info.action_dim
    print("Max_action", args.max_action)

    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv([
        lambda: gym.make(args.task,
                         world_name="21_05",
                         use_lppos=False,
                         use_predator=True,
                         max_step=30,
                         time_step=0.25,
                         reward_function=cwg.Reward({"puffed": -100, "finished": 10}))
        for _ in range(args.test_num)
    ])

    net = QRDQN(
        obs_dim=args.state_dim,
        action_shape=args.action_shape,
        num_quantiles=args.num_quantiles,
        device=args.device,
    )
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)
    policy: DiscreteCQLPolicy = DiscreteCQLPolicy(
        model=net,
        optim=optim,
        action_space=env.action_space,
        discount_factor=args.gamma,
        num_quantiles=args.num_quantiles,
        estimation_step=args.n_step,
        target_update_freq=args.target_update_freq,
        min_q_weight=args.min_q_weight,
    ).to(args.device)

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "cql"
    log_name = os.path.join(args.task, args.algo_name, now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    logger = TensorboardLogger(writer)

    def save_best_fn(policy: BasePolicy) -> None:
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards: float) -> bool:
        # Stop training if mean_rewards exceed a certain threshold
        # default threshold is none
        return False


    replay_buffer = load_buffer(args.expert_data)
    print(replay_buffer.obs.shape)
    print(type(replay_buffer.obs))
    print(replay_buffer.obs_next.shape)
    print(type(replay_buffer.obs_next))
    print(replay_buffer.act.shape)
    print(type(replay_buffer.act))
    print(replay_buffer.rew.shape)
    print(type(replay_buffer.rew))
    print(replay_buffer.done.shape)
    print(type(replay_buffer.done))
    print(type(replay_buffer.terminated))
    print(replay_buffer.terminated[:10])  # Print the first few elements to inspect their values

    # trainer
    result = OfflineTrainer(
        policy=policy,
        buffer=replay_buffer,
        test_collector=test_collector,
        max_epoch=args.epoch,
        step_per_epoch=args.update_per_epoch,
        episode_per_test=args.test_num,
        batch_size=args.batch_size,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        save_checkpoint_fn = None,
        logger=logger,
    ).run()
    pprint.pprint(result)


if __name__ == "__main__":
    test_cql()