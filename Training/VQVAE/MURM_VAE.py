import rlkit.util.hyperparameter as hyp
from rlkit.launchers.MyVAETrainer import MYVAETrainer
from rlkit.launchers.arglauncher import run_variants
from rlkit.torch.vae.vq_vae_trainer import VQ_VAETrainer
from rlkit.torch.jang.vqvae import train_vqvae
import rlkit.torch.pytorch_util as ptu

from rlkit.torch.vae.vq_vae import VQ_VAE
from rlkit.torch.vae.vq_vae import VQVAE2 as VQ_VAE2
from rlkit.torch.vae.vq_vae import VAE

image_train_data = '/media/jang/jang/0ubuntu/image_dataset/Combined_For_VQVAE/Combined_Active_Images.npy'
image_test_data = '/media/jang/jang/0ubuntu/image_dataset/Combined_For_VQVAE/Combined_Active_Images.npy'

ptu.set_gpu_mode(True)

if __name__ == "__main__":
    variant = dict(

        path_loader_kwargs=dict(
            delete_after_loading=True,
            recompute_reward=True,
        ),

        train_vae_kwargs=dict(
            imsize=128,
            beta=0.25, #1
            beta_schedule_kwargs=dict(
                x_values=(0, 250),
                y_values=(0, 100),
            ),
            num_epochs=1501,
            embedding_dim=64,
            dump_skew_debug_plots=False,
            decoder_activation='sigmoid',
            use_linear_dynamics=False,
            generate_vae_dataset_kwargs=dict(
                N=1000,
                n_random_steps=2,
                test_p=.9,
                dataset_path={'train': image_train_data,
                              'test': image_test_data,
                              },
                augment_data=False,
                use_cached=False,
                show=False,
                oracle_dataset=False,
                oracle_dataset_using_set_to_goal=False,
                non_presampled_goal_img_is_garbage=False,
                delete_after_loading=True,
                random_rollout_data=True,
                random_rollout_data_set_to_goal=True,
                conditional_vae_dataset=True,
                save_trajectories=False,
                enviorment_dataset=False,
                tag="ccrig_tuning_orig_network",
            ),
            vae_trainer_class=VQ_VAETrainer,
            vae_class=VQ_VAE,
            vae_kwargs=dict(
                input_channels=3,
                imsize=128,
            ),
            algo_kwargs=dict(
                key_to_reconstruct='x_t',
                start_skew_epoch=5000,
                is_auto_encoder=False,
                batch_size=128,
                lr=3e-4,  #1e-3,
                skew_config=dict(
                    method='vae_prob',
                    power=0,
                ),
                weight_decay=0, # VAl = 0
                skew_dataset=False,
                priority_function_kwargs=dict(
                    decoder_distribution='gaussian_identity_variance',
                    sampling_method='importance_sampling',
                   # num_latents_to_sample=10,
                ),
                use_parallel_dataloading=False,
            ),

            save_period=50,
        ),

        train_model_func=train_vqvae,

        launcher_config=dict(
            unpack_variant=True,
            region='South-Korea-Yonsei',
        ),
    )

    search_space = {
        #"seed": range(2), # Running Number of Variants
        'seed': [1],
    }
    sweeper = hyp.DeterministicHyperparameterSweeper(
        search_space, default_parameters=variant,
    )
    variants = []
    for variant in sweeper.iterate_hyperparameters():
        variants.append(variant)

    run_variants(MYVAETrainer, variants, run_id=0)
