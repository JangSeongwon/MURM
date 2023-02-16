import roboverse
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
parser.add_argument("--name", type=str, default='singleview')
# parser.add_argument("--name", type=str, default='multiview1')
# parser.add_argument("--name", type=str, default='multiview2')
parser.add_argument("--num_episodes", type=int, default=500)
parser.add_argument("--num_timesteps", type=int, default=250)
parser.add_argument("--video_save", type=int, default=0, help="Set to zero for no video saving")
args = parser.parse_args()

path = "/media/jang/jang/0ubuntu/demos_dataset/64/SingleView/demo_singleview_final/"
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
renderer_active = EnvRenderer_active(init_camera=None, **renderer_kwargs)

env = InsertImageEnv(state_env, renderer1=renderer, renderer2=renderer_active)
imlength = env.obs_img_dim * env.obs_img_dim * 3

success = 0
returns = 0
act_dim = env.action_space.shape[0]
# print(act_dim)
num_datasets = 0
demo_dataset = []
avg_tasks_done = 0

for j in tqdm(range(args.num_episodes)):
    env.demo_reset()

    trajectory = {
        'observations': [],
        'next_observations': [],
        'actions': np.zeros((args.num_timesteps, act_dim), dtype=np.float64),
        'rewards': np.zeros((args.num_timesteps), dtype=np.float64),
        'terminals': np.zeros((args.num_timesteps), dtype=np.uint8),
        'agent_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
        'env_infos': np.zeros((args.num_timesteps), dtype=np.uint8),
    }
    images = []
    images_active = []

    for i in range(args.num_timesteps):
        img = np.uint8(env.render_obs())
        img_active = np.uint8(env.render_obs_active())

        observation = env.get_observation()
        action = env.get_demo_action()
        #print('Saving Action', action)
        next_observation, reward, done, info = env.step(action)

        # if i == 1:
        #     print('3 4 3 12288 12288', observation.keys())
        #     print(next_observation.keys())
        #     for a in observation.values():
        #         print('observation', a.shape)
        #     # print(observation)
            # for a in next_observation.values():
            #     print('next_observation', a.shape)

        # print('observation = 64*64*3 12288', observation['image_global_observation'])
        # print('observation', observation['image_active_observation'])

        # Obs before action
        trajectory['observations'].append(observation)
        # Action given as delta_pos
        trajectory['actions'][i, :] = action
        # Obs after action
        trajectory['next_observations'].append(next_observation)
        trajectory['rewards'][i] = reward

        #Checking with videos (Global & Active)
        if args.video_save:
            img = env.render_obs()
            images.append(img)
        if args.video_save:
            img_active = env.render_obs_active()
            images_active.append(img_active)

    if reward == -1:
        print('checking final reward', reward)

    demo_dataset.append(trajectory)
    avg_tasks_done += env.done

    if ((j + 1) % 500) == 0:
        curr_name = demo_data_save_path + '_{0}.pkl'.format(num_datasets)
        file = open(curr_name, 'wb')
        pkl.dump(demo_dataset, file)
        file.close()
        demo_dataset = []

        num_datasets += 1

    if args.video_save:
        # if j % 30 == 0:
        roboverse.utils.save_video('{}/{}_global.avi'.format(path, j), images)
        roboverse.utils.save_video('{}/{}_active.avi'.format(path, j), images_active)

print('Demo Success Rate: {}'.format(avg_tasks_done / args.num_episodes))
