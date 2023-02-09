import os
from absl import app
from absl import flags

from roboverse.envs.murm_env import MURMENV
from roboverse.envs.murm_env_m1 import MURMENV_m1
from roboverse.envs.murm_env_m2 import MURMENV_m2

from rlkit.learning.murm import murm_experiment

import rlkit.util.hyperparameter as hyp
from rlkit.demos.source.encoder_dict_to_mdp_path_loader import EncoderDictToMDPPathLoader
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.sac.policies import GaussianPolicy
from rlkit.torch.sac.policies import GaussianTwoChannelCNNPolicy
from rlkit.torch.networks.cnn import TwoChannelCNN
from rlkit.torch.networks.cnn import ConcatTwoChannelCNN

from rlkit.learning.ptp import process_args
from rlkit.utils import arg_util
from rlkit.utils.logging import logger as logging

flags.DEFINE_string('name', None, '')
flags.DEFINE_string('base_log_dir', None, '')
flags.DEFINE_bool('local', True, '')
flags.DEFINE_bool('gpu', True, '')
flags.DEFINE_bool('save_pretrained', True, '')
flags.DEFINE_bool('debug', False, '')
flags.DEFINE_bool('script', False, '')
flags.DEFINE_multi_string('arg_binding', None, 'Variant binding to pass through.')
FLAGS = flags.FLAGS

def get_paths():
    data_path = '/media/jang/jang/0ubuntu/'
    demo_paths = [
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_1.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_2.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_3.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_4.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_5.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_6.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_7.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/Singleview_demos_100_8.pkl', obs_dict=True, is_demo=True, use_latents=True),

                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_final/Singleview_demos_100_1.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_final/Singleview_demos_100_2.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_final/Singleview_demos_100_3.pkl', obs_dict=True, is_demo=True, use_latents=True),
                # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_final/Singleview_demos_100_4.pkl', obs_dict=True, is_demo=True, use_latents=True),

                  dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_state_as_robot/1313.pkl', obs_dict=True, is_demo=True, use_latents=True),
                  # dict(path=data_path + 'demos_dataset/SingleView/demo_singleview_final/1414.pkl', obs_dict=True, is_demo=True, use_latents=True),
                  ]
    logging.info('data_path: %s', data_path)
    return data_path, demo_paths

def vqvae_assign(data_path):
    vqvae_path = os.path.join(data_path, 'Vae_Model/')
    return vqvae_path

def view_assign():
    #TODO: Assign camera view
    camera = 0
    if camera == 0:
        viewpoint = 'murm'
    elif camera == 1:
        viewpoint = 'g'
    elif camera == 2:
        viewpoint = 'a'
    else:
        print('No viewpoint selected')
        exit()
    return viewpoint

def env_class_assign():
    env_class = MURMENV
    # env_class = MURMENV_m1
    # env_class = MURMENV_m2
    return env_class

def get_default_variant(data_path, demo_paths, vqvae_path, viewpoint):

    default_variant = dict(
        pretrained_vqvae_path=vqvae_path,
        method_name='murm',
        MURM_view=viewpoint,

        imsize=128,
        env_kwargs=dict(
            test_env=True,
        ),
        policy_class=GaussianPolicy,
        policy_kwargs=dict(
            hidden_sizes=[256, 256, 256, 256],
            std=0.15,
            max_log_std=-1,
            min_log_std=-2,
            std_architecture='shared',
            output_activation=None,
        ),
        qf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),
        vf_kwargs=dict(
            hidden_sizes=[256, 256],
        ),

        env_type='SingleView',

        trainer_kwargs=dict(
            discount=0.995,
            policy_lr=3E-4,
            qf_lr=3E-4,
            reward_scale=1,

            policy_weight_decay=1e-4,
            q_weight_decay=0,

            reward_transform_kwargs=dict(m=1, b=0),
            terminal_transform_kwargs=None,

            beta=0.1,
            quantile=0.9,
            clip_score=100,

            min_value=None,
            max_value=None,
        ),

        max_path_length=400,
        algo_kwargs=dict(
            batch_size=256,
            start_epoch=-500,
            num_epochs=551,

            #TODO: Epoch steps Fix
            num_eval_steps_per_epoch=2000, #TODO: 2000: 400 per episode -> so 5 times eval
            num_expl_steps_per_train_loop=2000,
            num_trains_per_train_loop=1000,
            num_online_trains_per_train_loop=2000,
            min_num_steps_before_training=4000,

            eval_epoch_freq=5,
            offline_expl_epoch_freq=5,
        ),

        replay_buffer_kwargs=dict(
            fraction_next_context=0.1,
            fraction_future_context=0.6,
            fraction_foresight_context=0.0,
            fraction_perturbed_context=0.0,
            fraction_distribution_context=0.0,
            max_future_dt=None,
            max_size=int(1E6),
        ),

        online_offline_split=True,

        reward_kwargs=dict(
            reward_type='sparse',
            epsilon=2.0,
            epsilon_murm=1.0,
        ),

        online_offline_split_replay_buffer_kwargs=dict(
            online_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.6,
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_future_dt=None,
                max_size=int(4E5),
            ),
            offline_replay_buffer_kwargs=dict(
                fraction_next_context=0.1,
                fraction_future_context=0.9,
                fraction_foresight_context=0.0,
                fraction_perturbed_context=0.0,
                fraction_distribution_context=0.0,
                max_future_dt=None,
                max_size=int(6E5),
            ),
            sample_online_fraction=0.6
        ),

        observation_key='latent_observation',
        observation_keys=['latent_observation'],
        goal_key='latent_desired_goal',

        save_video=True,
        expl_save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),
        eval_save_video_kwargs=dict(
            save_video_period=10,
            pad_color=0,
        ),

        # reset_keys_map=dict(
        #     image_global_observation='initial_latent_state',
        #     image_active_observation='initial_latent_state_active',
        # ),

        path_loader_class=EncoderDictToMDPPathLoader,
        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
            demo_paths=demo_paths,
            split_max_steps=None,
        ),

        renderer_kwargs=dict(
            create_image_format='HWC',
            output_image_format='CWH',
            flatten_image=True,
            width=128,
            height=128,
        ),

        add_env_demos=False,
        add_env_offpolicy_data=False,

        load_demos=True,

        evaluation_goal_sampling_mode='given_latent',
        exploration_goal_sampling_mode='given_latent',
        training_goal_sampling_mode='given_latent',

        scripted_goals=None,
        # expl_reset_interval=0,

        launcher_config=dict(
            unpack_variant=True,
            region='South-Korea-Yonsei',
        ),
        logger_config=dict(
            snapshot_mode='gap',
            snapshot_gap=50,
        ),

        trainer_type='iql',
        network_version=None,

        pretrained_rl_path='', #'/media/jang/jang/0ubuntu/pretrained_rl/it0.pt',
        eval_seeds=14,
        # num_demos=20,
        num_video_columns=5,
        save_paths=False,

    )

    return default_variant


def get_search_space():
    ########################################
    # Search Space
    ########################################
    search_space = {

        'env_type': ['SingleView', 'Wall', 'RandomBox'],

        # Load up existing policy/q-network/value network vs train a new one
        'use_pretrained_rl_path': [False],

        # For only finetuning, set start_epoch=0.
        # 'algo_kwargs.start_epoch': [0],

        'trainer_kwargs.bc': [False],  # Run BC experiment
        # Reset environment every 'reset_interval' episodes
        'reset_interval': [1],

        # Training Hyperparameters
        'trainer_kwargs.beta': [0.01],

        # Overrides currently beta with beta_online during finetuning
        'trainer_kwargs.use_online_beta': [False],
        'trainer_kwargs.beta_online': [0.01],

        # Anneal beta every 'anneal_beta_every' by 'anneal_beta_by until
        # 'anneal_beta_stop_at'
        'trainer_kwargs.use_anneal_beta': [False],
        'trainer_kwargs.anneal_beta_every': [20],
        'trainer_kwargs.anneal_beta_by': [.05],
        'trainer_kwargs.anneal_beta_stop_at': [.0001],

        # If True, use pretrained reward classifier. If False, use epsilon.
        'reward_kwargs.use_pretrained_reward_classifier_path': [False],

        'trainer_kwargs.quantile': [0.9],

        'trainer_kwargs.use_online_quantile': [False],
        'trainer_kwargs.quantile_online': [0.99],

        # Network Parameters
        # Concatenate gripper position and rotation into network input
        'use_gripper_observation': [False],

        # Goals
        'use_both_ground_truth_and_affordance_expl_goals': [False],
        # 'affordance_sampling_prob': [1],
        'ground_truth_expl_goals': [True],
        'only_not_done_goals': [False],
    }

    return search_space


def process_variant(variant, data_path, env_class):  # NOQA
    # Error checking
    assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['eval_epoch_freq'] == 0  # NOQA
    if variant['algo_kwargs']['start_epoch'] < 0:
        assert variant['algo_kwargs']['start_epoch'] % variant['algo_kwargs']['offline_expl_epoch_freq'] == 0  # NOQA
    if variant['use_pretrained_rl_path']:
        assert variant['algo_kwargs']['start_epoch'] == 0
    if variant['trainer_kwargs']['use_online_beta']:
        assert variant['trainer_kwargs']['use_anneal_beta'] is False
    env_type = variant['env_type']

    ########################################
    # Set the eval_goals.
    ########################################
    full_open_close_str = ''
    if 'eval_seeds' in variant.keys():
        eval_seed_str = f"_seed{variant['eval_seeds']}"
    else:
        eval_seed_str = ''

    ########################################
    # Goal sampling modes.
    ########################################
    # variant['presampled_goal_kwargs']['eval_goals'] = eval_goals
    # variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size'] = min(  # NOQA
    #     int(6E5), int(500*75*variant['num_demos']))
    # variant['online_offline_split_replay_buffer_kwargs']['online_replay_buffer_kwargs']['max_size'] = min(  # NOQA
    #     int(4/6 * 500*75*variant['num_demos']),
    #     int(1E6 - variant['online_offline_split_replay_buffer_kwargs']['offline_replay_buffer_kwargs']['max_size']))  # NOQA

    # if variant['use_both_ground_truth_and_affordance_expl_goals']:
        # variant['exploration_goal_sampling_mode'] = (
        #     'conditional_vae_prior_and_not_done_presampled_images')
        #variant['training_goal_sampling_mode'] = 'presample_latents'
        # variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        # variant['presampled_goal_kwargs']['expl_goals_kwargs']['affordance_sampling_prob'] = variant['affordance_sampling_prob']  # NOQA
    # elif variant['ground_truth_expl_goals']:
        # variant['exploration_goal_sampling_mode'] = 'presampled_images'
        #variant['training_goal_sampling_mode'] = 'presampled_images'
        # variant['presampled_goal_kwargs']['expl_goals'] = eval_goals
        # variant['presampled_goal_kwargs']['training_goals'] = eval_goals

    # if variant['only_not_done_goals']:
    #     _old_mode = 'presampled_images'
    #     _new_mode = 'not_done_presampled_images'
    #
    #     if variant['training_goal_sampling_mode'] == _old_mode:
    #         variant['training_goal_sampling_mode'] = _new_mode
    #     if variant['exploration_goal_sampling_mode'] == _old_mode:
    #         variant['exploration_goal_sampling_mode'] = _new_mode
    #     if variant['evaluation_goal_sampling_mode'] == _old_mode:
    #         variant['evaluation_goal_sampling_mode'] = _new_mode

    ########################################
    # Environments.
    ########################################
    variant['env_class'] = env_class

    ########################################
    # Gripper Observation.
    ########################################
    # if variant['use_gripper_observation']:
    #
    #     variant['observation_keys'] = [
    #         'latent_observation',
    #         'gripper_state_observation']
    #     for demo_path in variant['path_loader_kwargs']['demo_paths']:
    #         demo_path['use_gripper_obs'] = True


    ########################################
    # Misc.
    ########################################
    if variant['reward_kwargs']['reward_type'] in ['sparse']:
        variant['trainer_kwargs']['max_value'] = 0.0
        variant['trainer_kwargs']['min_value'] = -1. / (
            1. - variant['trainer_kwargs']['discount'])

    if 'std' in variant['policy_kwargs']:
        if variant['policy_kwargs']['std'] <= 0:
            variant['policy_kwargs']['std'] = None


def main(_):
    data_path, demo_paths = get_paths()
    vqvae_path = vqvae_assign(data_path)
    viewpoint = view_assign()
    env_class = env_class_assign()

    default_variant = get_default_variant(data_path, demo_paths, vqvae_path, viewpoint)
    search_space = get_search_space()

    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space,
        default_parameters=default_variant,
    )

    logging.info('arg_binding: ')
    logging.info(FLAGS.arg_binding)

    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variant = arg_util.update_bindings(variant,
                                           FLAGS.arg_binding,
                                           check_exist=True)
        process_variant(variant, data_path, env_class)
        a= variant['env_type']
        print(a)

        variants.append(variant)

    run_variants(murm_experiment,
                 variants,
                 run_id=0,
                 process_args_fn=process_args)


if __name__ == '__main__':
    app.run(main)
