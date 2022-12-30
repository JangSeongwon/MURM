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
parser.add_argument("--name", type=str, default='singleview')
# parser.add_argument("--name", type=str, default='multiview1')
# parser.add_argument("--name", type=str, default='multiview2')
parser.add_argument("--num_episodes", type=int, default=300) # Analogies made 50K transitions (300 episodes=60K T) "--num_trajectories"
parser.add_argument("--num_timesteps", type=int, default=300) # 450 Making videos For Graduation exam #250 for 9 boxes env

# 185 timestep, 300 epi == 55k transitions

parser.add_argument("--video_save", type=int, default=1, help="Set to zero for no video saving")
args = parser.parse_args()

path = "/home/jang/data/data_collection/dataset/"
demo_data_save_path = path + args.name + "_demos"
gImage_save_path = path + args.name + "_global_images.npy"
aImage_save_path = path + args.name + "_active_images.npy"

a = roboverse.register_bullet_environments

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
# print(act_dim)
num_datasets = 0
demo_dataset = []
Global_image_dataset = {
    'observations': np.zeros((args.num_episodes, args.num_timesteps, imlength), dtype=np.uint8),
    # Saved in [[[ img_in_num (imlength)] *timestep] *num of trajectories]
    # 'object': [],
    'initial env': np.zeros((args.num_episodes, imlength), dtype=np.uint8),
    # Saving initial obs of the environment for each trajectory
}
Active_image_dataset = {
    'observations': np.zeros((args.num_episodes, args.num_timesteps, imlength), dtype=np.uint8),
    # Saved in [[[ img_in_num (imlength)] *timestep] *num of trajectories]
    # 'object': [],
    'initial env': np.zeros((args.num_episodes, imlength), dtype=np.uint8),
    # Saving initial obs of the environment for each trajectory
}

avg_tasks_done = 0
for j in tqdm(range(args.num_episodes)):
    env.demo_reset()
    # Initial Image Saving
    Global_image_dataset['initial env'][j, :] = np.uint8(env.render_obs().transpose()).flatten()
    Active_image_dataset['initial env'][j, :] = np.uint8(env.render_obs_active().transpose()).flatten()

    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float64),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float64),
    }

    images = []
    images_active = []

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        img_active = np.uint8(env.render_obs_active())
        # All images Savings
        Global_image_dataset['observations'][j, i, :] = img.transpose().flatten()
        Active_image_dataset['observations'][j, i, :] = img_active.transpose().flatten()

        observation = env.get_observation()

        action = env.get_demo_action()
        #print('Saving Action', action)
        next_observation, reward, done, info = env.step(action)
        #print('reward savings = -1', reward)

        # Obs before action
        trajectory['observations'].append(observation)
        # Action given as delta_pos
        trajectory['actions'][i, :] = action
        # Obs after action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

        #print('Checked to be 0 when success'.reward)

        #Checking with videos (Global & Active)
        # if args.video_save:
        #     img = env.render_obs()
        #     images.append(img)
        # if args.video_save:
        #     img_active = env.render_obs_active()
        #     images_active.append(img_active)
    print('reward', reward)
    demo_dataset.append(trajectory)
    avg_tasks_done += env.done

    if ((j + 1) % 500) == 0:
        curr_name = demo_data_save_path + '_{0}.pkl'.format(num_datasets)
        file = open(curr_name, 'wb')
        pkl.dump(demo_dataset, file)
        file.close()

        num_datasets += 1
        demo_dataset = []

    # if args.video_save:
    #     if j % 10 == 0:
    #         roboverse.utils.save_video('{}/{}_global.avi'.format(path, j), images)
    #         roboverse.utils.save_video('{}/{}_active.avi'.format(path, j), images_active)

print('Demo Success Rate: {}'.format(avg_tasks_done / args.num_episodes))
np.save(gImage_save_path, Global_image_dataset)
np.save(aImage_save_path, Active_image_dataset)

