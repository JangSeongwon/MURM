import os.path as osp
import time
import numpy as np
from torch.utils import data
from rlkit.util.io import load_local_or_remote_file

def train_vqvae(variant):
    vqvae = train_vae(variant)
    return vqvae

def train_vae(variant, return_data=False):
    from rlkit.util.ml_util import PiecewiseLinearSchedule, ConstantSchedule
    from rlkit.torch.vae.conv_vae import ConvVAE
    import rlkit.torch.vae.conv_vae as conv_vae
    from rlkit.torch.vae.vae_trainer import ConvVAETrainer
    from rlkit.core import logger
    import rlkit.torch.pytorch_util as ptu
    from rlkit.pythonplusplus import identity
    import torch

    beta = variant["beta"]
    representation_size = variant.get("representation_size",
        variant.get("latent_sizes", variant.get("embedding_dim", None)))
    variant['algo_kwargs']['num_epochs'] = variant['num_epochs']
    generate_vae_dataset_fctn = variant.get('generate_vae_data_fctn',
                                            generate_vae_dataset)
    variant['generate_vae_dataset_kwargs']['batch_size'] = variant['algo_kwargs']['batch_size']
    train_dataset, test_dataset, info = generate_vae_dataset_fctn(
        variant['generate_vae_dataset_kwargs'])

    logger.save_extra_data(info)
    logger.get_snapshot_dir()
    if 'beta_schedule_kwargs' in variant:
        beta_schedule = PiecewiseLinearSchedule(
            **variant['beta_schedule_kwargs'])
    else:
        beta_schedule = None
    if 'context_schedule' in variant:
        schedule = variant['context_schedule']
        if type(schedule) is dict:
            context_schedule = PiecewiseLinearSchedule(**schedule)
        else:
            context_schedule = ConstantSchedule(schedule)
        variant['algo_kwargs']['context_schedule'] = context_schedule
    if variant.get('decoder_activation', None) == 'sigmoid':
        decoder_activation = torch.nn.Sigmoid()
    else:
        decoder_activation = identity
    architecture = variant['vae_kwargs'].get('architecture', None)
    if not architecture and variant.get('imsize') == 84:
        architecture = conv_vae.imsize84_default_architecture
    elif not architecture and variant.get('imsize') == 48:
        architecture = conv_vae.imsize48_default_architecture
    variant['vae_kwargs']['architecture'] = architecture
    variant['vae_kwargs']['imsize'] = variant.get('imsize')

    vae_class = variant.get('vae_class', ConvVAE)
    model = vae_class(representation_size, decoder_output_activation=decoder_activation, **variant['vae_kwargs'])

    model.to(ptu.device)

    #print('model currently vqvae', model)
    vae_trainer_class = variant.get('vae_trainer_class', ConvVAETrainer)
    #print('VAE class ', vae_trainer_class)

    trainer = vae_trainer_class(model, beta=beta,
                       beta_schedule=beta_schedule,
                       **variant['algo_kwargs'])

    save_period = variant['save_period']

    dump_skew_debug_plots = variant.get('dump_skew_debug_plots', False)

    #######################################  TRAINING PROCESS ############################################

    for epoch in range(variant['num_epochs']):
        print('epoch & save_period',epoch, save_period)
        #IMAGE SHOWING PROCESS
        # should_save_imgs = (epoch % save_period == 0)
        should_save_imgs = (epoch % 1 == 0)
        trainer.train_epoch(epoch, train_dataset)

        print('Train Epoch GO')

        #trainer.test_epoch(epoch, test_dataset)

        if should_save_imgs:
            trainer.dump_reconstructions(epoch)
            trainer.dump_samples(epoch)
            if dump_skew_debug_plots:
                trainer.dump_best_reconstruction(epoch)
                trainer.dump_worst_reconstruction(epoch)
                trainer.dump_sampling_histogram(epoch)

        stats = trainer.get_diagnostics()
        for k, v in stats.items():
            logger.record_tabular(k, v)
        logger.dump_tabular()
        trainer.end_epoch(epoch)

        # Save the Model in .pt
        if epoch % 100 == 0:
            logger.save_itr_params(epoch, model)

    logger.save_extra_data(model, 'model', mode='pickle')

    if return_data:
        return model, train_dataset, test_dataset

    return model


def generate_vae_dataset(variant):

    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs',None)
    env_id = variant.get('env_id', None)
    N = variant.get('N', 10000)
    batch_size = variant.get('batch_size', 128)
    test_p = variant.get('test_p', 0.9)
    use_cached = variant.get('use_cached', True)
    imsize = variant.get('imsize', 84)
    num_channels = variant.get('num_channels', 3)
    show = variant.get('show', False)
    init_camera = variant.get('init_camera', None)
    dataset_path = variant.get('dataset_path', None)
    augment_data = variant.get('augment_data', False)
    data_filter_fn = variant.get('data_filter_fn', lambda x: x)
    delete_after_loading = variant.get('delete_after_loading', False)
    oracle_dataset_using_set_to_goal = variant.get('oracle_dataset_using_set_to_goal', False)
    random_rollout_data = variant.get('random_rollout_data', False)
    random_rollout_data_set_to_goal = variant.get('random_rollout_data_set_to_goal', True)
    random_and_oracle_policy_data=variant.get('random_and_oracle_policy_data', False)
    random_and_oracle_policy_data_split=variant.get('random_and_oracle_policy_data_split', 0)
    policy_file = variant.get('policy_file', None)
    n_random_steps = variant.get('n_random_steps', 100)
    vae_dataset_specific_env_kwargs = variant.get('vae_dataset_specific_env_kwargs', None)
    save_file_prefix = variant.get('save_file_prefix', None)
    non_presampled_goal_img_is_garbage = variant.get('non_presampled_goal_img_is_garbage', None)

    conditional_vae_dataset = variant.get('conditional_vae_dataset', False)
    use_env_labels = variant.get('use_env_labels', False)
    use_linear_dynamics = variant.get('use_linear_dynamics', False)
    enviorment_dataset = variant.get('enviorment_dataset', False)
    save_trajectories = variant.get('save_trajectories', False)
    save_trajectories = save_trajectories or use_linear_dynamics or conditional_vae_dataset
    tag = variant.get('tag', '')

    assert N % n_random_steps == 0, "Fix N/horizon or dataset generation will fail"

    from rlkit.util.io import load_local_or_remote_file
    from rlkit.data_management.dataset  import (
        TrajectoryDataset, ImageObservationDataset, InitialObservationDataset,
        EnvironmentDataset, ConditionalDynamicsDataset, InitialObservationNumpyDataset,
        InfiniteBatchLoader, InitialObservationNumpyJitteringDataset
    )
    info = {}
    use_test_dataset = False

    if dataset_path is not None:
        if isinstance(dataset_path, dict):

            if type(dataset_path['train']) == str:

                print('TRAIN DATASET PATH HERE')
                dataset = load_local_or_remote_file(dataset_path['train'], delete_after_loading=delete_after_loading)
                dataset = dataset.item()

            if type(dataset_path['test']) == str:

                print('TEST DATASET PATH HERE')
                test_dataset = load_local_or_remote_file(dataset_path['test'], delete_after_loading=delete_after_loading)
                test_dataset = test_dataset.item()

            N = dataset['observations'].shape[0] * dataset['observations'].shape[1]
            #print('N = 1700 ',N)
            n_random_steps = dataset['observations'].shape[1]
            use_test_dataset = True
    else:
        pass

    info['train_labels'] = []
    info['test_labels'] = []

    dataset = data_filter_fn(dataset)

    if conditional_vae_dataset:
        num_trajectories = N // n_random_steps
        n = int(num_trajectories * test_p)
        indices = np.arange(num_trajectories)
        np.random.shuffle(indices)
        train_i, test_i = indices[:n], indices[n:]

        if augment_data:
            dataset_class = InitialObservationNumpyJitteringDataset
        else:
            dataset_class = InitialObservationNumpyDataset

        if 'env' not in dataset:
            dataset['env'] = dataset['observations'][:, 0]
        if use_test_dataset and ('env' not in test_dataset):
            test_dataset['env'] = test_dataset['observations'][:, 0]


        if use_test_dataset:
            train_dataset = dataset_class({
                'observations': dataset['observations'],
                'env': dataset['env']
            })

            test_dataset = dataset_class({
                'observations': test_dataset['observations'],
                'env': test_dataset['env']
            })
        else:
            train_dataset = dataset_class({
                'observations': dataset['observations'][train_i, :, :],
                'env': dataset['env'][train_i, :]
            })

            test_dataset = dataset_class({
                'observations': dataset['observations'][test_i, :, :],
                'env': dataset['env'][test_i, :]
            })


        train_batch_loader_kwargs = variant.get(
            'train_batch_loader_kwargs',
            dict(batch_size=batch_size, num_workers=0, )
        )
        test_batch_loader_kwargs = variant.get(
            'test_batch_loader_kwargs',
            dict(batch_size=batch_size, num_workers=0, )
        )

        train_data_loader = data.DataLoader(train_dataset,
            shuffle=True, drop_last=True, **train_batch_loader_kwargs)
        test_data_loader = data.DataLoader(test_dataset,
            shuffle=True, drop_last=True, **test_batch_loader_kwargs)

        train_dataset = InfiniteBatchLoader(train_data_loader)
        #print('HERE')
        test_dataset = InfiniteBatchLoader(test_data_loader)
    else:
        exit()

    return train_dataset, test_dataset, info
