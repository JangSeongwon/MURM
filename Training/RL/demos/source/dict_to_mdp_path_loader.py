import glob
import numpy as np
import copy

from rlkit.data_management.path_builder import PathBuilder
from rlkit.torch.core import np_to_pytorch_batch
from rlkit.util.io import (
    load_local_or_remote_file,
    sync_down_folder,
    get_absolute_path
)
from rlkit.utils.real_utils import filter_step_fn


def split_demo(demo, max_steps):
    if max_steps is None:
        return [demo]

    else:
        new_demo = []

        key, value = next(iter(demo.items()))
        horizon = len(value)

        t = np.random.randint(0, min(max_steps, horizon - max_steps))

        while True:
            new_demo_t = {}
            new_t = t + max_steps
            if new_t >= horizon:
                break

            for key, value in demo.items():
                if key in ['object_name', 'skill_id']:
                    new_demo_t[key] = value
                else:
                    new_demo_t[key] = value[t:new_t]

            t = new_t

            new_demo.append(new_demo_t)

        return new_demo


class DictToMDPPathLoader:
    """
    Path loader for that loads obs-dict demonstrations
    into a Trainer with EnvReplayBuffer
    """

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            env=None,
            reward_fn=None,
            demo_paths=None,  # list of dicts
            demo_train_split=0.9,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            # Return true to add path, false to ignore it
            data_filter_fn=lambda x: True,
            split_max_steps=None,
            filter_step_fn=filter_step_fn,
            min_action_value=None,
            action_round_thresh=None,
            min_path_length=None,
            **kwargs
    ):
        self.trainer = trainer
        self.delete_after_loading = delete_after_loading
        self.add_demos_to_replay_buffer = add_demos_to_replay_buffer
        self.demo_train_split = demo_train_split
        self.demo_data_split = demo_data_split
        self.replay_buffer = replay_buffer
        self.demo_train_buffer = demo_train_buffer
        self.demo_test_buffer = demo_test_buffer
        self.data_filter_fn = data_filter_fn

        self.env = env
        self.reward_fn = reward_fn

        self.demo_paths = [] if demo_paths is None else demo_paths

        self.bc_num_pretrain_steps = bc_num_pretrain_steps
        self.q_num_pretrain_steps = q_num_pretrain_steps
        self.demo_trajectory_rewards = []

        self.env_info_key = env_info_key
        self.obs_key = obs_key
        self.recompute_reward = recompute_reward
        self.load_terminals = load_terminals

        self.trainer.replay_buffer = self.replay_buffer
        self.trainer.demo_train_buffer = self.demo_train_buffer
        self.trainer.demo_test_buffer = self.demo_test_buffer

        self.split_max_steps = split_max_steps

        self.filter_step_fn = filter_step_fn
        self.min_action_value = min_action_value
        self.action_round_thresh = action_round_thresh

        self.min_path_length = min_path_length

    def preprocess(self, observation):
        print('Not use')
        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def load_demos(self):
        # Off policy
        for demo_path in self.demo_paths:
            self.load_demo_path(**demo_path)

    def load_demo_path(self,  
                       path,
                       is_demo,
                       obs_dict,
                       train_split=None,
                       data_split=None,
                       sync_dir=None,
                       use_latents=True,
                       use_gripper_obs=False):
        print('loading demo path', path)

        if sync_dir is not None:
            sync_down_folder(sync_dir)
            paths = glob.glob(get_absolute_path(path))
            # print('Not here')
        else:
            paths = [path]

        data = []
        for filename in paths:
            data_i = load_local_or_remote_file(
                filename,
                delete_after_loading=self.delete_after_loading)
            data_i = list(data_i)

            if self.split_max_steps:
                new_data_i = []
                for j in range(len(data_i)):
                    data_i_j = split_demo(data_i[j],
                                          max_steps=self.split_max_steps)
                    new_data_i.extend(data_i_j)
                data_i = new_data_i

            data.extend(data_i)

        if train_split is None:
            train_split = self.demo_train_split
            # print('train split', train_split)

        if data_split is None:
            data_split = self.demo_data_split
            # print('data split', data_split)

        M = int(len(data) * train_split * data_split)
        N = int(len(data) * data_split)
        print('using', M, 'trajectories for training')
        # print('using', N, 'data for testing')

        if self.add_demos_to_replay_buffer:
            for path in data[:M]:
                self.load_path(path,
                               self.replay_buffer,
                               obs_dict=obs_dict,
                               use_latents=use_latents,
                               use_gripper_obs=use_gripper_obs)

        if is_demo:
            print('Please Check DEMO BUFFER IN HERE')
            if self.demo_train_buffer:
                print('demo_train_buffer')
                for path in data[:M]:
                    self.load_path(path,
                                   self.demo_train_buffer,
                                   obs_dict=obs_dict,
                                   use_latents=use_latents,
                                   use_gripper_obs=use_gripper_obs)

            if self.demo_test_buffer:
                print('demo_test_buffer')
                for path in data[M:N]:
                    self.load_path(path,
                                   self.demo_test_buffer,
                                   obs_dict=obs_dict,
                                   use_latents=use_latents,
                                   use_gripper_obs=use_gripper_obs)

    def get_batch_from_buffer(self, replay_buffer):
        batch = replay_buffer.random_batch(self.bc_batch_size)
        batch = np_to_pytorch_batch(batch)
        return batch
