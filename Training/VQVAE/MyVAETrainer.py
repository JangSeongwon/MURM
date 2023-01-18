
from rlkit.util.io import load_local_or_remote_file
from rlkit.torch.jang.vqvae import train_vae

def MYVAETrainer(
        train_vae_kwargs,
        path_loader_kwargs=None,
        train_model_func=train_vae,):

    model = train_model_func(train_vae_kwargs)
    path_loader_kwargs['model'] = model

    print('-----------------Finished Training VQVAE for MURM-----------------')



