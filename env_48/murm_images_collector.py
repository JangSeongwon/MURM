import roboverse
import random
import numpy as np
import pybullet as p
import pickle as pkl
from tqdm import tqdm
from roboverse.utils.renderer import EnvRenderer, InsertImageEnv, EnvRenderer_active
from roboverse.bullet.misc import quat_to_deg 
import os
from PIL import Image
import argparse
# physicsClient = p.connect(p.GUI)

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='task1')
# parser.add_argument("--name", type=str, default='task2')
# parser.add_argument("--name", type=str, default='task3')
parser.add_argument("--num_episodes", type=int, default=1000)
parser.add_argument("--num_timesteps", type=int, default=100)
parser.add_argument("--video_save", type=int, default=0, help="Set to zero for no video saving")
args = parser.parse_args()

path = "/media/jang/jang/0ubuntu/image_dataset/48/"
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
renderer_active = EnvRenderer_active(init_camera=None, **renderer_kwargs)

env = InsertImageEnv(state_env, renderer1=renderer, renderer2=renderer_active)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
avg_tasks_done = 0

def image_data():
    Global_image_dataset = {
        'observations': np.zeros((args.num_episodes, args.num_timesteps, imlength), dtype=np.uint8),
        # Saved in [[[ img_in_num (imlength)] *timestep] *num of trajectories]
    }
    #print('npy format', Global_image_dataset)
    Active_image_dataset = {
        'observations': np.zeros((args.num_episodes, args.num_timesteps, imlength), dtype=np.uint8),
        # Saved in [[[ img_in_num (imlength)] *timestep] *num of trajectories]
    }
    return Global_image_dataset, Active_image_dataset

Global_image_dataset, Active_image_dataset = image_data()

for j in tqdm(range(args.num_episodes)):

    env.demo_reset()
    images = []
    images_active = []
    if j % 5 == 0:
        x = 1
        n = random.uniform(0, 0.1)
        n = round(n, 2)
        print(n)
    else:
        x = 0

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        img_active = np.uint8(env.render_obs_active())
        # All images Savings
        Global_image_dataset['observations'][j, i, :] = img.transpose().flatten()
        Active_image_dataset['observations'][j, i, :] = img_active.transpose().flatten()

        if x == 0:
            action = env.get_demo_action()
        else:
            action = env.get_demo_action_noisy(n)
        next_observation, reward, done, info = env.step(action)

        #Checking with videos (Global & Active)
        if args.video_save:
            img = env.render_obs()
            images.append(img)
        if args.video_save:
            img_active = env.render_obs_active()
            images_active.append(img_active)

    if reward == -1:
        print('reward', reward)
    avg_tasks_done += env.done

    if args.video_save:
        roboverse.utils.save_video('{}/{}_global.avi'.format(path, j), images)
        roboverse.utils.save_video('{}/{}_active.avi'.format(path, j), images_active)

print('Demo Success Rate: {}'.format(avg_tasks_done / args.num_episodes))
np.save(gImage_save_path, Global_image_dataset)
np.save(aImage_save_path, Active_image_dataset)
