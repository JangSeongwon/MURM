import copy
import numpy as np

import rlkit.torch.pytorch_util as ptu
from rlkit.demos.source.dict_to_mdp_path_loader import DictToMDPPathLoader
from rlkit.util.io import load_local_or_remote_file
from rlkit.data_management.path_builder import PathBuilder
from roboverse.bullet.misc import quat_to_deg


class EncoderDictToMDPPathLoader(DictToMDPPathLoader):

    def __init__(
            self,
            trainer,
            replay_buffer,
            demo_train_buffer,
            demo_test_buffer,
            model=None,
            model_global=None,
            model_active=None,
            model_path=None,
            reward_fn=None,
            compare_reward_fn=None,
            MURM_view=None,
            env=None,
            demo_paths=[],  # list of dicts
            normalize=False,
            demo_train_split=1,
            demo_data_split=1,
            add_demos_to_replay_buffer=True,
            condition_encoding=False,
            bc_num_pretrain_steps=0,
            bc_batch_size=64,
            bc_weight=1.0,
            rl_weight=1.0,
            q_num_pretrain_steps=0,
            weight_decay=0,
            eval_policy=None,
            recompute_reward=False,
            object_list=None,
            env_info_key=None,
            obs_key=None,
            load_terminals=True,
            delete_after_loading=False,
            # Return true to add path, false to ignore it
            data_filter_fn=lambda x: True,
            split_max_steps=None,
            **kwargs
    ):
        super().__init__(trainer=trainer,
                         replay_buffer=replay_buffer,
                         demo_train_buffer=demo_train_buffer,
                         demo_test_buffer=demo_test_buffer,
                         demo_paths=demo_paths,
                         demo_train_split=demo_train_split,
                         demo_data_split=demo_data_split,
                         add_demos_to_replay_buffer=add_demos_to_replay_buffer,
                         bc_num_pretrain_steps=bc_num_pretrain_steps,
                         bc_batch_size=bc_batch_size,
                         bc_weight=bc_weight,
                         rl_weight=rl_weight,
                         q_num_pretrain_steps=q_num_pretrain_steps,
                         weight_decay=weight_decay,
                         eval_policy=eval_policy,
                         recompute_reward=recompute_reward,
                         env_info_key=env_info_key,
                         obs_key=obs_key,
                         load_terminals=load_terminals,
                         delete_after_loading=delete_after_loading,
                         data_filter_fn=data_filter_fn,
                         split_max_steps=split_max_steps,
                         **kwargs)

        if model is None:
            exit()
        else:
            assert model_path is None
            if MURM_view == 'murm':
                self.model_global = model_global
                self.model_active = model_active
            else:
                self.model = model
                # print(self.model)

        self.condition_encoding = condition_encoding
        self.reward_fn = reward_fn
        self.compare_reward_fn = compare_reward_fn
        self.normalize = normalize
        self.object_list = object_list
        self.env = env
        self.murm = MURM_view
        self.counting_loaded_demos = 0

    def preprocess(self, observation, use_latents=True, use_gripper_obs=False):

        # print('SEE Previous version', observation)
        observation = copy.deepcopy(observation[:-1])

        # b = np.stack([observation[0]['image_global_observation']])
        # print('SEE Previous version', b.shape)

        if self.murm == 'murm':
            images_global = np.stack([observation[i]['image_global_observation']
                              for i in range(len(observation))])

            images_active = np.stack([observation[i]['image_active_observation']
                              for i in range(len(observation))])
            latents_global = self.model['vqvae'].encode_np(images_global)
            latents_active = self.model['vqvae'].encode_np(images_active)
            print('LATENTS G/A', latents_global.shape, latents_active.shape)

        elif self.murm == 'g':
            images = np.stack([observation[i]['image_global_observation']
                                      for i in range(len(observation))])
            latents = self.model['vqvae'].encode_np(images)
            # print('LATENTS G', latents.shape)

        elif self.murm == 'a':
            images = np.stack([observation[i]['image_active_observation']
                                      for i in range(len(observation))])
            latents = self.model['vqvae'].encode_np(images)
            # print('LATENTS A', latents.shape)

        else:
            exit()

        for i in range(len(observation)):
            if self.murm == 'murm':
                # observation[i]['initial_latent_state'] = latents_global[0]
                observation[i]['latent_observation'] = latents_global[i]
                observation[i]['latent_desired_goal'] = latents_global[-1]
                # observation[i]['initial_latent_state_active'] = latents_active[0]
                observation[i]['latent_observation_active'] = latents_active[i]
                observation[i]['latent_desired_goal_active'] = latents_active[-1]

            elif self.murm == 'g':
                # observation[i]['initial_latent_state'] = latents[0]
                observation[i]['latent_observation'] = latents[i]
                observation[i]['latent_desired_goal'] = latents[-1]
                # observation[i]['initial_latent_state_active'] = [0]

            elif self.murm == 'a':
                # observation[i]['initial_latent_state'] = latents[0]
                observation[i]['latent_observation'] = latents[i]
                observation[i]['latent_desired_goal'] = latents[-1]
                # observation[i]['initial_latent_state_active'] = [0] #Just needed for murm code to run
            else:
                exit()

            if use_latents:
                del observation[i]['image_global_observation']
                del observation[i]['image_active_observation']
            else:
                observation[i]['initial_image_observation'] = images_global[0]
                observation[i]['image_observation'] = images_global[i]
                observation[i]['image_desired_goal'] = images_global[-1]

        # print('Checking final output version', observation)
        # a = np.stack([observation[0]['latent_observation']])
        # print('SEE Later version', a.shape)

        return observation

    def preprocess_array_obs(self, observation):
        new_observations = []
        for i in range(len(observation)):
            new_observations.append(dict(observation=observation[i]))
        return new_observations

    def encode(self, obs):
        if self.normalize:
            return ptu.get_numpy(
                self.model.encode(ptu.from_numpy(obs) / 255.0))
        return ptu.get_numpy(self.model.encode(ptu.from_numpy(obs)))

    def load_path(self,
                  path,
                  replay_buffer,
                  obs_dict=None,
                  use_latents=True,
                  use_gripper_obs=False):

        # print('Correct Latent Changing')

        # Filter data #
        if not self.data_filter_fn(path):
            return

        rewards = []
        compare_rewards = []
        path_builder = PathBuilder()
        H = min(len(path['observations']), len(path['actions'])) - 1

        if obs_dict:
            traj_obs = self.preprocess(
                path['observations'],
                use_latents=use_latents,
                use_gripper_obs=use_gripper_obs)
            next_traj_obs = self.preprocess(
                path['next_observations'],
                use_latents=use_latents,
                use_gripper_obs=use_gripper_obs)
        else:
            traj_obs = self.preprocess_array_obs(
                path['observations'])
            next_traj_obs = self.preprocess_array_obs(
                path['next_observations'])

        for i in range(H):
            ob = traj_obs[i]
            next_ob = next_traj_obs[i]
            action = path['actions'][i]
            reward = path['rewards'][i]
            terminal = path['terminals'][i]
            if not self.load_terminals:
                terminal = np.zeros(terminal.shape)
            agent_info = path['agent_infos'][i]
            env_info = path['env_infos'][i]

            terminal = np.array([terminal]).reshape((1,))

            # print('recompute reward == True', self.recompute_reward)

            if self.recompute_reward:
                reward, terminal = self.reward_fn(ob, action, next_ob, next_ob)

            reward = np.array([reward]).flatten()
            rewards.append(reward)

            if self.recompute_reward and self.compare_reward_fn:
                compare_reward, _ = self.compare_reward_fn(
                    ob, action, next_ob, next_ob)
                compare_rewards.append(compare_reward)

            # print('obs', ob)
            # print('actions', action)

            path_builder.add_all(
                observations=ob,
                actions=action,
                rewards=reward,
                next_observations=next_ob,
                terminals=terminal,
                agent_infos=agent_info,
                env_infos=env_info,
            )

        self.counting_loaded_demos += 1
        print('demos loaded = ', self.counting_loaded_demos)

        self.demo_trajectory_rewards.append(rewards)
        path = path_builder.get_all_stacked()
        replay_buffer.add_path(path)
        print('length of obs and action', len(
            path['observations']), len(path['actions']))
        # print('actions', np.min(path['actions']), np.max(path['actions'])) #TODO: Due to Gripper: -1, 1 are min and max
        # print('rewards', np.min(rewards), np.max(rewards))
        # print('path sum rewards', sum(rewards), len(rewards))
        if self.compare_reward_fn:
            print('Min / Max rewards',
                  np.min(compare_rewards), np.max(compare_rewards))
            print('Sum rewards',
                  sum(compare_rewards), len(compare_rewards))
