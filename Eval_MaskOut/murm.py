import os.path as osp
from collections import OrderedDict  
from functools import partial
from scipy.stats import bernoulli

import numpy as np
import torch
from sklearn.manifold import TSNE
from gym.wrappers import ClipAction
import random
import rlkit.torch.pytorch_util as ptu
from rlkit.core import logger
from rlkit.data_management.contextual_replay_buffer import (
    #ContextualRelabelingReplayBuffer,
    RemapKeyFn,
)
from rlkit.data_management.online_offline_split_replay_buffer import (
    OnlineOfflineSplitReplayBuffer,
)
from rlkit.envs.contextual.goal_conditioned import (
    GoalDictDistributionFromMultitaskEnv,
    AddImageDistribution,
    PresampledPathDistribution,
)
from rlkit.envs.contextual.latent_distributions import (
    AmortizedConditionalPriorDistribution,
    PresampledPriorDistribution,
    ConditionalPriorDistribution,
    AmortizedPriorDistribution,
    AddDecodedImageDistribution,
    AddLatentDistribution,
    AddGripperStateDistribution,
    PriorDistribution,
    PresamplePriorDistribution,
    murmlatentgoalspace,
)
from rlkit.envs.encoder_wrappers import EncoderWrappedEnv
from rlkit.envs.gripper_state_wrapper import GripperStateWrappedEnv
from rlkit.envs.gripper_state_wrapper import process_gripper_state

from rlkit.envs.images import EnvRenderer
from rlkit.envs.images import EnvRenderer_active

from rlkit.envs.images import InsertImageEnv
from rlkit.demos.source.mdp_path_loader import MDPPathLoader
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.torch.networks import ConcatMlp, Mlp
from rlkit.torch.networks.cnn import ConcatCNN
from rlkit.torch.networks.cnn import ConcatTwoChannelCNN
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import MakeDeterministic
from rlkit.torch.sac.iql_trainer import IQLTrainer
from rlkit.torch.sac.sac import SACTrainer
from rlkit.torch.torch_rl_algorithm import TorchBatchRLAlgorithm
from rlkit.launchers.contextual.rig.rig_launcher import StateImageGoalDiagnosticsFn
from rlkit.launchers.contextual.util import get_gym_env
from rlkit.launchers.rl_exp_launcher_util import create_exploration_policy
from rlkit.util.io import load_local_or_remote_file
from rlkit.visualization.video import RIGVideoSaveFunction
from rlkit.visualization.video import save_paths as save_paths_fn
from rlkit.samplers.data_collector.contextual_path_collector import ContextualPathCollector
from rlkit.samplers.rollout_functions import contextual_rollout

from rlkit.envs.contextual_env import ContextualEnv
from rlkit.learning.contextual_replay_buffer import ContextualRelabelingReplayBuffer
from rlkit.utils.logging import logger as logging
from rlkit.utils import io_util


class RewardFn:
    def __init__(self,
                 MURM_view,
                 env,
                 obs_type='latent',
                 reward_type='dense',
                 epsilon=3.0,
                 epsilon_murm=2.5,
                 ):

        if obs_type == 'latent':
            self.obs_key = 'latent_observation'
            self.obs_key_murm = 'latent_observation_murm'
            self.goal_key = 'latent_desired_goal'
            self.goal_key_murm = 'latent_desired_goal_murm'

        elif obs_type == 'state':
            self.obs_key = 'state_observation'
            self.goal_key = 'state_desired_goal'

        self.MURM_view = MURM_view
        self.env = env
        self.reward_type = reward_type
        self.epsilon = epsilon
        self.epsilon_murm = epsilon_murm

    def process(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    def __call__(self, states, actions, next_states, contexts):
        s = self.process(next_states[self.obs_key])
        # print('s', next_states)
        '''len = 5120 in demo buffer'''
        # print('contexts', contexts)

        if self.MURM_view == 'murm':
            # print('Getting MURM goals')
            s2 = self.process(next_states[self.obs_key_murm])
            # print('s2', s2, s2.shape)
            c = self.process(contexts[self.goal_key])
            # print('c', c, c.shape)
            c2 = self.process(contexts[self.goal_key_murm])
            # print('c2', c2, c2.shape)
        else:
            # print('self goal key', self.goal_key)
            c = self.process(contexts[self.goal_key])

        terminal = np.zeros((s.shape[0], ), dtype=np.uint8)
        # print('Reward Type', self.reward_type)

        if self.reward_type == 'dense':
            reward = -np.linalg.norm(s - c, axis=1)

        elif self.reward_type == 'sparse' and self.MURM_view == 'g':
            # print('epsilon', self.epsilon)
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            reward = success - 1

        elif self.reward_type == 'sparse' and self.MURM_view == 'a':
            # print('epsilon', self.epsilon)
            success = np.linalg.norm(s - c, axis=1) < self.epsilon
            success1 = np.linalg.norm(s - c, axis=1)
            # print('Reward Distance', success1)
            reward = success - 1

        elif self.reward_type == 'sparse' and self.MURM_view == 'murm':
            x = np.linalg.norm(s - c, axis=1) < self.epsilon
            y = np.linalg.norm(s2 - c2, axis=1) < self.epsilon_murm
            xx = np.linalg.norm(s - c, axis=1)
            yy = np.linalg.norm(s2 - c2, axis=1)
            # print('xx yy', xx, yy)
            # reward = x*y - 2
            # print('x, y', x, y)
            # print('x', x.size)
            if 0 == x.size and y.size == 0:
                reward = np.empty(0)
            elif x.size == 256 and y.size == 256:
                success = []
                for i in range(256):
                    p = [int(x[i])]
                    # print('p', p)
                    success = np.append(success, p, axis=0)
                # print(success.size, success)
                success1 = []
                for i in range(256):
                    q = [int(y[i])]
                    # print('q', q)
                    success1 = np.append(success1, q, axis=0)
                # print(success1.size, success1)
                reward = success + success1 - 2

                # print('reward', reward)
            elif x.size == 153 or x.size == 103 or y.size == 153 or y.size == 103:
                success = []
                if x.size == 153:
                    num = 153
                else:
                    num = 103
                for i in range(num):
                    p = [int(x[i])]
                    # print('p', p)
                    success = np.append(success, p, axis=0)
                # print(success.size, success)
                success1 = []
                for i in range(num):
                    q = [int(y[i])]
                    # print('q', q)
                    success1 = np.append(success1, q, axis=0)
                # print(success1.size, success1)
                reward = success + success1 - 2
            else:
                success = int(x)
                success1 = int(y)
                reward = np.array([success + success1 - 2])

            # print(self.epsilon, self.epsilon_murm)
            # print('MURM reward', reward)

        else:
            raise ValueError(self.reward_type)
        return reward, terminal


def process_args(variant):
    # Maybe adjust the arguments for debugging purposes.
    if variant.get('debug', False):
        # variant['max_path_length'] = 5
        # variant['num_presample'] = 50
        # variant['num_presample'] = 32
        variant.get('algo_kwargs', {}).update(dict(
            batch_size=32,
            start_epoch=-1,
            # start_epoch=0,
            num_epochs=5,
            num_eval_steps_per_epoch=variant['max_path_length'],  # * 5,
            num_expl_steps_per_train_loop=variant['max_path_length'],  # * 5,
            num_trains_per_train_loop=2,
            num_online_trains_per_train_loop=2,
            min_num_steps_before_training=2,
        ))
        variant['num_video_columns'] = 1
        variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = int(5E2)
        variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = int(5E2)
        demo_paths = variant['path_loader_kwargs'].get('demo_paths', [])
        if len(demo_paths) > 1:
            variant['path_loader_kwargs']['demo_paths'] = [demo_paths[0]]

#
# def add_gripper_state_obs(
#     rollout
# ):
#     def wrapper(*args, **kwargs):
#         paths = rollout(*args, **kwargs)
#         for i in range(paths['observations'].shape[0]):
#             d = paths['observations'][i]
#             d['gripper_state_observation'] = process_gripper_state(
#                 d['state_observation'])
#             d['gripper_state_desired_goal'] = process_gripper_state(
#                 d['state_desired_goal'])
#
#         for i in range(paths['next_observations'].shape[0]):
#             d = paths['next_observations'][i]
#             d['gripper_state_observation'] = process_gripper_state(
#                 d['state_observation'])
#             d['gripper_state_desired_goal'] = process_gripper_state(
#                 d['state_desired_goal'])
#         return paths
#     return wrapper


def murm_experiment(
        max_path_length,
        qf_kwargs,
        vf_kwargs,
        trainer_kwargs,
        replay_buffer_kwargs,
        online_offline_split_replay_buffer_kwargs,
        policy_kwargs,
        algo_kwargs,
        online_offline_split=False,
        policy_class=None,
        env_id=None,
        env_class=None,
        env_kwargs=None,
        reward_kwargs=None,
        encoder_wrapper=EncoderWrappedEnv,
        observation_key='latent_observation',
        murm_observation_adding_key='latent_observation_murm',
        observation_keys=['latent_observation'],
        observation_key_reward_fn=None,
        goal_key='latent_desired_goal',
        murm_goal_adding_key='latent_desired_goal_murm',
        goal_key_reward_fn=None,
        state_observation_key='state_observation',
        robot_state_observation_key='robot_state_observation',
        gripper_observation_key='gripper_state_observation',
        state_goal_key='state_desired_goal',
        image_goal_key='image_desired_goal',
        image_init_key='initial_image_observation',
        gripper_goal_key='gripper_state_desired_goal',
        reset_keys_map=None,
        use_gripper_observation=False,

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=None,
        env_demo_path='',
        env_offpolicy_data_path='',

        debug=False,
        epsilon=1.0,
        exploration_policy_kwargs=None,
        evaluation_goal_sampling_mode=None,
        exploration_goal_sampling_mode=None,
        training_goal_sampling_mode=None,

        add_env_demos=False,
        add_env_offpolicy_data=False,
        save_paths=True,
        load_demos=False,
        pretrain_policy=False,
        pretrain_rl=False,
        save_pretrained_algorithm=False,

        trainer_type='iql',
        network_version=None,

        # Video parameters
        save_video=True,
        save_video_pickle=False,
        expl_save_video_kwargs=None,
        eval_save_video_kwargs=None,
        renderer_kwargs=None,
        imsize=48,
        pretrained_vae_path='',
        pretrained_rl_path='',
        use_pretrained_rl_path=False,
        input_representation='',
        goal_representation='',
        presampled_goal_kwargs=None,
        presampled_goals_path='',
        num_presample=50,
        num_video_columns=8,
        init_camera=None,
        qf_class=ConcatMlp,
        vf_class=Mlp,
        env_type=None,  # For plotting
        seed=None,
        multiple_goals_eval_seeds=None,
        expl_reset_interval=0,
        expl_contextual_env_kwargs=None,
        eval_contextual_env_kwargs=None,
        MURM_view=None,
        pretrained_vqvae_path='',
        **kwargs
):
    # Kwarg Definitions
    if exploration_policy_kwargs is None:
        exploration_policy_kwargs = {}
    if reset_keys_map is None:
        reset_keys_map = {}
    if presampled_goal_kwargs is None:
        presampled_goal_kwargs = \
            {'eval_goals': '', 'expl_goals': '', 'training_goals': ''}
    if path_loader_kwargs is None:
        path_loader_kwargs = {}
    if not expl_save_video_kwargs:
        expl_save_video_kwargs = {}
    if not eval_save_video_kwargs:
        eval_save_video_kwargs = {}
    if not renderer_kwargs:
        renderer_kwargs = {}

    if env_type == 'trial2':
        reward_kwargs['epsilon_murm'] = 1.0

    elif env_type == 'trial3':
        reward_kwargs['epsilon_murm'] = 2.0
        # reward_kwargs['epsilon_murm'] = 1.0
        # policy_kwargs['hidden_sizes'] = [128, 128, 128, 128],
        # qf_kwargs['hidden_sizes'] = [128, 128]
        # vf_kwargs['hidden_sizes'] = [128, 128]

    elif env_type == 'trial4':
        algo_kwargs['start_epoch'] = -200
        algo_kwargs['num_epoch'] = 151
        a = online_offline_split_replay_buffer_kwargs['offline_replay_buffer_kwargs']
        a['fraction_next_context'] = 0.1
        a['fraction_future_context'] = 0.5

    elif env_type == 'trial4':
        algo_kwargs['start_epoch'] = -200
        algo_kwargs['num_epoch'] = 201


    # Enviorment Wrapping
    logging.info('Creating the environment...')
    torch.cuda.current_device()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('GPU', device)

    """For video production"""
    renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
    # renderer_active = EnvRenderer_active(init_camera=None, **renderer_kwargs)

    if goal_key_reward_fn is not None:
        distrib_goal_key = goal_key_reward_fn
    else:
        distrib_goal_key = goal_key

    def contextual_env_distrib_and_reward(
        env_id,
        env_class,
        env_kwargs,
        encoder_wrapper,
        goal_sampling_mode,
        reward_kwargs,
        model,

    ):
        # if MURM_view == 'murm':
        #     vqvae = model['vqvae']
        #     vqvae_add_murm = model['vqvae_murm']
        #     print('murm model', model)
        #
        # elif MURM_view == 'g':
        #     vqvae = model['vqvae']
        #     vqvae_add_murm = None
        #     print('global model', model)
        #
        # else:
        #     vqvae = model['vqvae']
        #     vqvae_add_murm = None
        #     print('active model', model)

        state_env = get_gym_env(
            env_id,
            env_class=env_class,
            env_kwargs=env_kwargs,
        )
        state_env = ClipAction(state_env)
        """Not using codes roboverse"""
        renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
        renderer_active = EnvRenderer_active(init_camera=None, **renderer_kwargs)

        img_env = InsertImageEnv(state_env, renderer1=renderer, renderer2=renderer_active)

        if MURM_view == 'murm':
            step_keys = dict(image_global_observation='latent_observation',
                             image_active_observation='latent_observation_murm',)
        elif MURM_view == 'g':
            step_keys = dict(image_global_observation='latent_observation')
        elif MURM_view == 'a':
            step_keys = dict(image_active_observation='latent_observation')
        else:
            exit()

        encoded_env = encoder_wrapper(
            MURM_view,
            img_env,
            model=model,
            step_keys_map=step_keys,
            reset_keys_map=reset_keys_map,
        )

        if goal_sampling_mode == 'given_latent':
            latent_goal_distribution = murmlatentgoalspace(
                model=model,
                env=state_env,
                key=distrib_goal_key,
                rep=model['vqvae'].representation_size,
                murm_view=MURM_view
            )
            diagnostics = state_env.get_contextual_diagnostics
        else:
            raise ValueError

        reward_fn = RewardFn(
            MURM_view,
            state_env,
            **reward_kwargs
        )

        if not isinstance(diagnostics, list):
            contextual_diagnostics_fns = [diagnostics]
        else:
            contextual_diagnostics_fns = diagnostics

        """ In env, goal is sampled from distribution """
        env = ContextualEnv(
                encoded_env,
                context_distribution=latent_goal_distribution,
                reward_fn=reward_fn,
                observation_key=observation_key,
                contextual_diagnostics_fns=[diagnostics] if not isinstance(
                    diagnostics, list) else diagnostics,
        )

        return env, latent_goal_distribution, reward_fn


    ######################################################################################
    ####################################  START  #########################################
    ######################################################################################
    code_global = 'g'
    code_active = 'a'
    code_murm = 'murm'

    if MURM_view == 'murm':
        model = io_util.load_model(pretrained_vqvae_path, code=code_murm)
        path_loader_kwargs['model'] = model
        # print('MURM VQVAE Two Model', model, model_murm)

    elif MURM_view == 'g':
        model = io_util.load_model(pretrained_vqvae_path, code=code_global)
        path_loader_kwargs['model'] = model

    elif MURM_view == 'a':
        model = io_util.load_model(pretrained_vqvae_path, code=code_active)
        path_loader_kwargs['model'] = model
    else:
        exit()

    # Environment Definitions
    expl_env_kwargs = env_kwargs.copy()
    expl_env_kwargs['expl'] = True
    exploration_goal_sampling_mode = evaluation_goal_sampling_mode

    logging.info('Preparing the [exploration] env and contextual distrib...')
    logging.info('sampling mode: %r', exploration_goal_sampling_mode)

    expl_env, expl_context_distrib, expl_reward = (
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            encoder_wrapper,
            exploration_goal_sampling_mode,
            reward_kwargs=reward_kwargs,
            model=model,

        ))

    logging.info('Preparing the [evaluation] env and contextual distrib...')
    logging.info('Preparing the eval env and contextual distrib...')
    logging.info('sampling mode: %r', evaluation_goal_sampling_mode)

    eval_env, eval_context_distrib, eval_reward = (contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            encoder_wrapper,
            evaluation_goal_sampling_mode,
            reward_kwargs=reward_kwargs,
            model=model,
        ))

    compare_reward_kwargs = reward_kwargs.copy()
    compare_reward_kwargs['reward_type'] = 'sparse'
    logging.info('Preparing the [training] env and contextual distrib...')
    logging.info('sampling mode: %r', training_goal_sampling_mode)

    _, training_context_distrib, compare_reward = (
        contextual_env_distrib_and_reward(
            env_id,
            env_class,
            env_kwargs,
            encoder_wrapper,
            training_goal_sampling_mode,
            reward_kwargs=compare_reward_kwargs,
            model=model,

        ))

    logging.info('Preparing the IQL code...')
    path_loader_kwargs['env'] = eval_env

    # IQL Code
    if add_env_demos:
        path_loader_kwargs['demo_paths'].append(env_demo_path)
    if add_env_offpolicy_data:
        path_loader_kwargs['demo_paths'].append(env_offpolicy_data_path)

    ''' Key Setting '''
    context_key = goal_key

    murm_context_key = murm_goal_adding_key
    murm_observation_key = murm_observation_adding_key
    # print('MURM context key', murm_context_key)
    # print('MURM observation key', murm_observation_key)

    if MURM_view == 'murm':
        obs_dim = (
                eval_env.observation_space.spaces[observation_key].low.size
                + eval_env.observation_space.spaces[context_key].low.size
                + eval_env.observation_space.spaces[murm_observation_key].low.size
                + eval_env.observation_space.spaces[murm_context_key].low.size
        )
        print('obs_dim', obs_dim)
    else:
        obs_dim = (
                eval_env.observation_space.spaces[observation_key].low.size
                + eval_env.observation_space.spaces[context_key].low.size
        )
        print('obs_dim', obs_dim, '////', eval_env.observation_space)

    action_dim = eval_env.action_space.low.size
    state_rewards = reward_kwargs.get('reward_type', 'dense') == 'highlevel'

    mapper_dict = {context_key: observation_key}

    obs_keys = [] #[observation_key]
    cont_keys = [context_key]
    cont_keys_to_save = []

    # if goal_key_reward_fn:
    #     print('Delete this part')
    #     mapper_dict[goal_key_reward_fn] = observation_key_reward_fn
    #     if goal_key_reward_fn not in obs_keys:
    #         obs_keys.append(observation_key_reward_fn)
    #     if goal_key_reward_fn not in cont_keys:
    #         cont_keys.append(goal_key_reward_fn)

    if MURM_view == 'murm':
        observation_keys.append(murm_observation_adding_key)
        mapper_dict = {context_key: observation_key, murm_context_key: murm_observation_key}
        cont_keys.append(murm_goal_adding_key)
    obs_keys.append(state_observation_key)
    # obs_keys.append(robot_state_observation_key) #TODO: fixed

    if state_rewards:
        mapper_dict[state_goal_key] = state_observation_key
        obs_keys.append(state_observation_key)
        cont_keys_to_save.append(state_goal_key)

    print('obs key checking', obs_keys, 'Training keys:', observation_keys)

    mapper = RemapKeyFn(mapper_dict)
    # print('mapper', mapper)

    # Replay Buffer
    def concat_context_to_obs(batch,
                              replay_buffer,
                              obs_dict,
                              next_obs_dict,
                              new_contexts):
        obs = batch['observations']
        # print('tuple?', obs, '//', obs[0], '//', obs[1])
        next_obs = batch['next_observations']

        if MURM_view == 'murm':
            batch['observations'] = np.concatenate([obs[0],
                                                    new_contexts['latent_desired_goal'],
                                                    obs[1],
                                                    new_contexts['latent_desired_goal_murm']], axis=1)
            batch['next_observations'] = np.concatenate([next_obs[0],
                                                         new_contexts['latent_desired_goal'],
                                                         next_obs[1],
                                                         new_contexts['latent_desired_goal_murm']], axis=1)

            if batch['observations'].size == 0:
                # print('pass')
                pass
            else:
                # print('BEFORE', batch['observations'])

                dropout1 = bernoulli.rvs(size=1, p=0.5)
                dropout2 = bernoulli.rvs(size=1, p=0.5)
                dropout1 = int(dropout1)
                dropout2 = int(dropout2)
                # print('dropout', dropout1, dropout2)
                if dropout1 == 0 and dropout2 == 0:
                    pass
                elif dropout1 == 0 and dropout2 == 1:
                    for i in range(256):
                        # if i % 2 == 0:
                        batch['observations'][i][1440:] = batch['observations'][i][1440:] * 0
                        batch['next_observations'][i][1440:] = batch['next_observations'][i][1440:] * 0
                        # print('NOW', batch['observations'][i])

                elif dropout1 == 1 and dropout2 == 0:
                    for i in range(256):
                        # if i % 2 == 0:
                        batch['observations'][i][:1440] = batch['observations'][i][:1440] * 0
                        batch['next_observations'][i][:1440] = batch['next_observations'][i][:1440] * 0
                        # print('NOW', batch['observations'][i])
                else:
                    pass
                # print('final', batch['observations'])

        else:
            if type(obs) is tuple:
                obs = np.concatenate(obs, axis=1)
            if type(next_obs) is tuple:
                next_obs = np.concatenate(next_obs, axis=1)
            if len(new_contexts.keys()) > 1:
                #     print('check', len(new_contexts.keys()))
                #     print('check 2', new_contexts['latent_desired_goal'])
                context = np.concatenate(tuple(new_contexts.values()), axis=1)
            else:
                context = batch[context_key]
            batch['observations'] = np.concatenate([obs, context], axis=1)
            batch['next_observations'] = np.concatenate([next_obs, context], axis=1)
        # print('batch see',batch)
        return batch

    online_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'online_replay_buffer_kwargs']
    offline_replay_buffer_kwargs = online_offline_split_replay_buffer_kwargs[
        'offline_replay_buffer_kwargs']

    # for rb_kwargs in [
    #         online_replay_buffer_kwargs,
    #         offline_replay_buffer_kwargs]:
    #     rb_kwargs['fraction_next_context'] = (
    #         replay_buffer_kwargs['fraction_next_context'])
    #     rb_kwargs['fraction_future_context'] = (
    #         replay_buffer_kwargs['fraction_future_context'])
    #     rb_kwargs['fraction_foresight_context'] = (
    #         replay_buffer_kwargs['fraction_foresight_context'])
    #     rb_kwargs['fraction_perturbed_context'] = (
    #         replay_buffer_kwargs['fraction_perturbed_context'])
    #     rb_kwargs['fraction_distribution_context'] = (
    #         replay_buffer_kwargs['fraction_distribution_context'])
    #     rb_kwargs['max_future_dt'] = (
    #         replay_buffer_kwargs['max_future_dt'])

    if online_offline_split:
        online_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(
                observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **online_replay_buffer_kwargs,
        )

        offline_replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(
                observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **offline_replay_buffer_kwargs
        )

        replay_buffer = OnlineOfflineSplitReplayBuffer(
            offline_replay_buffer,
            online_replay_buffer,
            **online_offline_split_replay_buffer_kwargs
        )

        logging.info('online_replay_buffer_kwargs: %r',
                     online_replay_buffer_kwargs)
        logging.info('offline_replay_buffer_kwargs: %r',
                     offline_replay_buffer_kwargs)
        logging.info('online_offline_split_replay_buffer_kwargs: %r',
                     online_offline_split_replay_buffer_kwargs)


    else:
        replay_buffer = ContextualRelabelingReplayBuffer(
            env=eval_env,
            context_keys=cont_keys,
            observation_keys_to_save=obs_keys,
            observation_key=(observation_key if observation_keys is None else None),
            observation_key_reward_fn=observation_key_reward_fn,
            observation_keys=observation_keys,
            context_distribution=training_context_distrib,
            sample_context_from_obs_dict_fn=mapper,
            reward_fn=eval_reward,
            post_process_batch_fn=concat_context_to_obs,
            context_keys_to_save=cont_keys_to_save,
            **replay_buffer_kwargs
        )


    if use_pretrained_rl_path:
        logging.info('Loading pretrained RL from: %s', pretrained_rl_path)
        rl_model_dict = load_local_or_remote_file(pretrained_rl_path)
        qf1 = rl_model_dict['trainer/qf1']
        qf2 = rl_model_dict['trainer/qf2']
        target_qf1 = rl_model_dict['trainer/target_qf1']
        target_qf2 = rl_model_dict['trainer/target_qf2']
        vf = rl_model_dict['trainer/vf']
        policy = rl_model_dict['trainer/policy']
        if 'std' in policy_kwargs and policy_kwargs['std'] is not None:
            policy.std = policy_kwargs['std']
            policy.log_std = np.log(policy.std)
    else:
        # Neural Network Architecture

        def create_qf():
            if qf_class is ConcatMlp:
                qf_kwargs['input_size'] = obs_dim + action_dim
            if qf_class is ConcatCNN or qf_class is ConcatTwoChannelCNN:
                qf_kwargs['added_fc_input_size'] = action_dim
            return qf_class(
                output_size=1,
                **qf_kwargs
            )

        qf1 = create_qf()
        qf2 = create_qf()
        # print('qf1', qf1)
        target_qf1 = create_qf()
        target_qf2 = create_qf()

        def create_vf():
            if vf_class is Mlp:
                vf_kwargs['input_size'] = obs_dim
            return vf_class(
                output_size=1,
                **vf_kwargs
            )
        vf = create_vf()

        if policy_class is GaussianPolicy:
            assert policy_kwargs['output_activation'] is None

        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )

    # Path Collectors
    path_collector_observation_keys = [
        observation_key, ] if observation_keys is None else observation_keys
    path_collector_context_keys_for_policy = [context_key, ]
    if MURM_view == 'murm':
        path_collector_observation_keys = [
            observation_key, murm_observation_adding_key, ] if observation_keys is None else observation_keys
        path_collector_context_keys_for_policy = [context_key, murm_goal_adding_key, ]

    def obs_processor(o):
        combined_obs = []
        if MURM_view == 'murm':
            # print('keys', observation_key, murm_observation_adding_key, goal_key, murm_goal_adding_key)
            combined_obs.append(o[observation_key])
            # print(o[goal_key])
            combined_obs.append(o[goal_key])
            combined_obs.append(o[murm_observation_adding_key])
            combined_obs.append(o[murm_goal_adding_key])
            # print(combined_obs)
        else:
            for k in path_collector_observation_keys:
                # print('checking keys finally = latent observation', path_collector_observation_keys)
                combined_obs.append(o[k])

            for k in path_collector_context_keys_for_policy:
                # print('checking keys finally = latent goals', path_collector_context_keys_for_policy)
                combined_obs.append(o[k])
            # print('combined obs', combined_obs, len(combined_obs))
        return np.concatenate(combined_obs, axis=0)

    def obs_processor_eval(o):
        random1 = [1, 2, 3]
        pick = random.choice(random1)
        combined_obs = []
        if pick == 3:
            if MURM_view == 'murm':
                dropout = bernoulli.rvs(size=1, p=0.5)
                dropout = int(dropout)
                if dropout == 0:
                    o[observation_key] = o[observation_key]*0
                    o[goal_key] = o[goal_key]*0
                else:
                    o[murm_observation_adding_key] = o[murm_observation_adding_key]*0
                    o[murm_goal_adding_key] = o[murm_goal_adding_key]*0

                # print('keys', observation_key, murm_observation_adding_key, goal_key, murm_goal_adding_key)
                combined_obs.append(o[observation_key])
                # print(o[goal_key])
                combined_obs.append(o[goal_key])
                combined_obs.append(o[murm_observation_adding_key])
                combined_obs.append(o[murm_goal_adding_key])
                # print(combined_obs)
            else:
                for k in path_collector_observation_keys:
                    o[k] = o[k]*0
                    # print('0 vector in eval', o[k])
                    # print('checking keys finally = latent observation', path_collector_observation_keys)
                    combined_obs.append(o[k])

                for k in path_collector_context_keys_for_policy:
                    o[k] = o[k] * 0
                    # print('checking keys finally = latent goals', path_collector_context_keys_for_policy)
                    combined_obs.append(o[k])
                # print('combined obs', combined_obs, len(combined_obs))
            return np.concatenate(combined_obs, axis=0)
        else:
            if MURM_view == 'murm':
                # print('keys', observation_key, murm_observation_adding_key, goal_key, murm_goal_adding_key)
                combined_obs.append(o[observation_key])
                # print(o[goal_key])
                combined_obs.append(o[goal_key])
                combined_obs.append(o[murm_observation_adding_key])
                combined_obs.append(o[murm_goal_adding_key])
                # print(combined_obs)
            else:
                for k in path_collector_observation_keys:
                    # print('checking keys finally = latent observation', path_collector_observation_keys)
                    combined_obs.append(o[k])

                for k in path_collector_context_keys_for_policy:
                    # print('checking keys finally = latent goals', path_collector_context_keys_for_policy)
                    combined_obs.append(o[k])
                # print('combined obs', combined_obs, len(combined_obs))
            return np.concatenate(combined_obs, axis=0)

    rollout = contextual_rollout

    eval_policy = policy

    eval_path_collector = ContextualPathCollector(
        eval_env,
        eval_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=obs_processor_eval,
        rollout=rollout,
    )

    expl_policy = create_exploration_policy(
        expl_env,
        policy,
        **exploration_policy_kwargs)

    expl_path_collector = ContextualPathCollector(
        expl_env,
        expl_policy,
        observation_keys=path_collector_observation_keys,
        context_keys_for_policy=path_collector_context_keys_for_policy,
        obs_processor=obs_processor,
        rollout=rollout,
    )

    if trainer_type == 'iql':
        if trainer_kwargs['use_online_beta']:
            if algo_kwargs['start_epoch'] == 0:
                trainer_kwargs['beta'] = trainer_kwargs['beta_online']

        if trainer_kwargs['use_online_quantile']:
            if algo_kwargs['start_epoch'] == 0:
                trainer_kwargs['quantile'] = trainer_kwargs['quantile_online']

    model['vf'] = vf
    model['qf1'] = qf1
    model['qf2'] = qf2
    # print('Model Check', model)

    # Algorithm
    trainer = IQLTrainer(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        vf=vf,
        **trainer_kwargs
    )

    algorithm = TorchBatchRLAlgorithm(
        trainer=trainer,
        exploration_env=expl_env,
        evaluation_env=eval_env,
        exploration_data_collector=expl_path_collector,
        evaluation_data_collector=eval_path_collector,
        replay_buffer=replay_buffer,
        max_path_length=max_path_length,
        **algo_kwargs
    )

    if trainer_type == 'iql':
        if trainer_kwargs['use_online_beta']:
            def switch_beta(self, epoch):
                if epoch == -1:
                    self.trainer.beta = trainer_kwargs['beta_online']
            algorithm.post_epoch_funcs.append(switch_beta)

        if trainer_kwargs['use_online_quantile']:
            def switch_quantile(self, epoch):
                if epoch == -1:
                    print('Switching quantile from %f to %f' % (
                        self.trainer.quantile,
                        trainer_kwargs['quantile_online']))
                    self.trainer.quantile = trainer_kwargs['quantile_online']
            algorithm.post_epoch_funcs.append(switch_quantile)

        elif trainer_kwargs['use_anneal_beta']:
            def switch_beta(self, epoch):
                if (epoch != algo_kwargs['start_epoch'] and
                        (epoch - algo_kwargs['start_epoch'])
                        % trainer_kwargs['anneal_beta_every'] == 0 and
                        self.trainer.beta * trainer_kwargs['anneal_beta_by']
                        >= trainer_kwargs['anneal_beta_stop_at']):
                    self.trainer.beta *= trainer_kwargs['anneal_beta_by']
            algorithm.post_epoch_funcs.append(switch_beta)

    algorithm.to(ptu.device)

    # Video Saving
    if save_video:
        # assert (num_video_columns * max_path_length <=
        #         algo_kwargs['num_expl_steps_per_train_loop'])

        expl_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'
        eval_save_video_kwargs['decode_image_goal_key'] = 'image_decoded_goal'  
        exp = 0
        eval = 1

        # expl_video_func = RIGVideoSaveFunction(
        #     model['vqvae'],
        #     expl_path_collector,
        #     'train',
        #     video=exp,
        #     image_goal_key=image_goal_key,
        #     rows=2,
        #     columns=num_video_columns,
        #     imsize=imsize,
        #     image_format=renderer.output_image_format,
        #     unnormalize=True,
        #     dump_pickle=save_video_pickle,
        #     dump_only_init_and_goal=True,
        #     **expl_save_video_kwargs
        # )
        # algorithm.post_train_funcs.append(expl_video_func)

        if algo_kwargs['num_eval_steps_per_epoch'] > 0:
            eval_video_func = RIGVideoSaveFunction(
                model['vqvae'],
                eval_path_collector,
                'eval',
                video=eval,
                image_goal_key=image_goal_key,
                rows=2,
                columns=num_video_columns,
                imsize=imsize,
                image_format=renderer.output_image_format,
                unnormalize=True,
                dump_pickle=save_video_pickle,
                dump_only_init_and_goal=True,
                **eval_save_video_kwargs
            )
            algorithm.post_train_funcs.append(eval_video_func)


    # IQL CODE
    if save_paths:
        algorithm.post_train_funcs.append(save_paths_fn)

    if online_offline_split:
        replay_buffer.set_online_mode(False)

    if load_demos:
        demo_train_buffer = None
        demo_test_buffer = None
        path_loader = path_loader_class(trainer,
                                        replay_buffer=replay_buffer,
                                        demo_train_buffer=demo_train_buffer,
                                        demo_test_buffer=demo_test_buffer,
                                        reward_fn=eval_reward,
                                        compare_reward_fn=compare_reward,
                                        MURM_view=MURM_view,
                                        **path_loader_kwargs
                                        )
        path_loader.load_demos()

    if save_pretrained_algorithm:
        p_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.p')
        pt_path = osp.join(logger.get_snapshot_dir(), 'pretrain_algorithm.pt')
        data = algorithm._get_snapshot()
        data['algorithm'] = algorithm
        torch.save(data, open(pt_path, 'wb'))
        torch.save(data, open(p_path, 'wb'))

    if online_offline_split:
        replay_buffer.set_online_mode(True)

    logging.info('Start training...')
    algorithm.train()
