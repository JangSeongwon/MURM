import roboverse
import numpy as np
import pybullet as p
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import math
import argparse

physicsClient = p.connect(p.GUI)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='jang')
parser.add_argument("--num_trajectories", type=int, default=4)
parser.add_argument("--num_timesteps", type=int, default=10)
parser.add_argument("--video_save", type=int, default=1, help="Set to zero for no video saving")
args = parser.parse_args()

path = "/home/jang/data/data_collection/dataset/"
demo_data_save_path = path + args.name + "_demos"
recon_data_save_path = path + args.name + "_images.npy"

state_env = roboverse.make('MURMENV-v0', object_subset='test')
imsize = state_env.obs_img_dim

renderer_kwargs=dict(
        create_image_format='HWC',
        output_image_format='CWH',
        width=imsize,
        height=imsize,
        flatten_image=True,)

renderer = EnvRenderer(init_camera=None, **renderer_kwargs)
env = InsertImageEnv(state_env, renderer=renderer)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
act_dim = env.action_space.shape[0]
num_datasets = 0
demo_dataset = []
recon_dataset = {
    'observations': np.zeros((args.num_trajectories, args.num_timesteps, imlength), dtype=np.uint8),
    'object': [],
    'env': np.zeros((args.num_trajectories, imlength), dtype=np.uint8),
}

avg_tasks_done = 0
for j in tqdm(range(args.num_trajectories)):
    env.demo_reset()
    recon_dataset['env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
    recon_dataset['object'].append(env.curr_object)
    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float64),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float64),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'object_name': env.curr_object,
    }

    images = []
    images_active = []

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        img_active = np.uint8(env.render_obs_active())
        recon_dataset['observations'][j, i, :] = img.transpose().flatten()

        observation = env.get_observation()

        action = env.get_demo_action()
        next_observation, reward, done, info = env.step(action)

        trajectory['observations'].append(observation)
        trajectory['actions'][i, :] = action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

        # Checking with videos (Global & Active)
        if args.video_save:
            img = env.render_obs()
            images.append(img)
        if args.video_save:
            img_active = env.render_obs_active()
            images_active.append(img_active)

    demo_dataset.append(trajectory)
    avg_tasks_done += env.done

    if ((j + 1) % 2) == 0:
        curr_name = demo_data_save_path + '_{0}.pkl'.format(num_datasets)
        file = open(curr_name, 'wb')
        pkl.dump(demo_dataset, file)
        file.close()

        num_datasets += 1
        demo_dataset = []

    if args.video_save:
        roboverse.utils.save_video('{}/{}_global.avi'.format(path, j), images)
    if args.video_save:
        roboverse.utils.save_video('{}/{}_active.avi'.format(path, j), images_active)

print('Success Rate: {}'.format(avg_tasks_done / args.num_trajectories))
np.save(recon_data_save_path, recon_dataset)

