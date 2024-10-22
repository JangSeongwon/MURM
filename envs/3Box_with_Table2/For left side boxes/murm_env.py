import roboverse
import os
import math as m
import time


import roboverse.bullet as bullet
import numpy as np
import pybullet as p
from gym.spaces import Box, Dict
from collections import OrderedDict
from roboverse.envs.panda_base import PandaBaseEnv
from roboverse.bullet.misc import load_obj, deg_to_quat, quat_to_deg, bbox_intersecting
from roboverse.envs.assets.bulletobjects.bullet_objects import loader, metadata
import os.path as osp
import importlib.util
import random
import pickle
import gym

quat_dict={'mug': [0, -1, 0, 1],'long_sofa': [0, 0, 0, 1],'camera': [-1, 0, 0, 0],
        'grill_trash_can': [0,0,1,1], 'beer_bottle': [0, 0, 1, -1], 'cube':[0,0,1,0] }

test_set = ['mug', 'long_sofa', 'camera', 'grill_trash_can', 'beer_bottle']

class MURMENV(PandaBaseEnv):

    def __init__(self,
                 goal_pos=(0,0,0),
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=True,
                 observation_mode='state',
                 #Image Dimension
                 obs_img_dim=1024, #sawyer=48
                 obs_img_dim_active=1024,
                 success_threshold=0.08,
                 transpose_image=False,
                 invisible_robot=False,
                 object_subset='all',
                 use_bounding_box=True,
                 random_color_p=1,
                 quat_dict=quat_dict,
                 task='goal_reaching',
                 test_env=False,
                 env_type=None,
                 DoF = 6,
                 *args,
                 **kwargs
                 ):
        """
        Grasping env with a single object
        :param goal_pos: xyz coordinate of desired goal
        :param reward_type: one of 'shaped', 'sparse'
        :param reward_min: minimum possible reward per timestep
        :param randomize: whether to randomize the object position or not
        :param observation_mode: state, pixels, pixels_debug
        :param obs_img_dim: image dimensions for the observations
        :param transpose_image: first dimension is channel when true
        :param invisible_robot: the robot arm is invisible when set to True
        """

        print('INIT')

        assert DoF in [3, 6, 7]
        assert task in ['goal_reaching', 'pickup', 'Pick and Place']
        print("Task Type: " + task)

        is_set = object_subset in ['test', 'train', 'all']
        is_list = type(object_subset) == list
        assert is_set or is_list

        self.goal_pos = np.asarray(goal_pos)
        self.quat_dict = quat_dict
        self._reward_type = reward_type
        self._reward_min = reward_min
        self._randomize = randomize
        self.pickup_eps = -0.3
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.random_color_p = random_color_p
        self.use_bounding_box = use_bounding_box
        self.object_subset = object_subset
        self.test_env = test_env
        self._ddeg_scale = 5
        self.task = task
        self.DoF = DoF


        if self.test_env:
            self.random_color_p = 0.0
            if object_subset is 'all':
                self.object_subset = ['grill_trash_can']

        # self.object_dict, self.scaling = self.get_object_info()
        self.curr_object = None
        self._object_position_low = (0.4, -0.2, 1.01)
        self._object_position_high = (0.8, 0.2, 1.8)
        self._fixed_object_position = np.array([.4, 0, 1.018])
        self.start_obj_ind = 4 if (self.DoF == 4) else 8
        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._success_threshold = success_threshold

        #Global Camera
        self.obs_img_dim = obs_img_dim #+.15
        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.8, 0, 1.5], distance=0.8,
            yaw=90, pitch=-20, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)
        #Active Camera
        self.obs_img_dim_active = obs_img_dim_active

        self.dt = 0.1
        super().__init__(*args, **kwargs)
        self._max_force = 100
        self._timeStep = 1. / 240.

    def random_goal_generation(self):
        list_floor = [1,2,3]
        floor_choice = random.choice(list_floor)
        list_shelf = [1,2,3]
        shelf_choice = random.choice(list_shelf)
        self.shelf_chosen = shelf_choice

        if floor_choice == 1:
            z_floor = 1.09
            if shelf_choice == 1:
                self._goal_low = np.array([-0.2, -0.8, z_floor])
                self._goal_high = np.array([0.2, -0.8, z_floor])
            elif shelf_choice == 2:
                self._goal_low = np.array([0.8, -0.2, z_floor])
                self._goal_high = np.array([0.8, 0.2, z_floor])
            else:
                self._goal_low = np.array([-0.2, 0.8, z_floor])
                self._goal_high = np.array([0.2, 0.8, z_floor])
        elif floor_choice == 2:
            z_floor = 1.28
            if shelf_choice == 1:
                self._goal_low = np.array([-0.2, -0.8, z_floor])
                self._goal_high = np.array([0.2, -0.8, z_floor])
            elif shelf_choice == 2:
                self._goal_low = np.array([0.8, -0.2, z_floor])
                self._goal_high = np.array([0.8, 0.2, z_floor])
            else:
                self._goal_low = np.array([-0.2, 0.8, z_floor])
                self._goal_high = np.array([0.2, 0.8, z_floor])
        elif floor_choice == 3:
            z_floor = 1.49
            if shelf_choice == 1:
                self._goal_low = np.array([-0.2, -0.8, z_floor])
                self._goal_high = np.array([0.2, -0.8, z_floor])
            elif shelf_choice == 2:
                self._goal_low = np.array([0.8, -0.2, z_floor])
                self._goal_high = np.array([0.8, 0.2, z_floor])
            else:
                self._goal_low = np.array([-0.2, 0.8, z_floor])
                self._goal_high = np.array([0.2, 0.8, z_floor])

        goal = np.random.uniform(low=self._goal_low, high=self._goal_high)

        return goal

    def reset(self, change_object=False):

        # Load Enviorment
        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        p.setGravity(0, 0, -9.8)

        self._load_table()
        self._floor = bullet.objects.marble_floor()
        self.goal_near = 0

        #Robot
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES | p.URDF_USE_INERTIA_FROM_FILE | p.URDF_USE_SELF_COLLISION
        self._panda = p.loadURDF(os.path.join('/home/jang/PycharmProjects/bullet env/roboverse/envs/assets/Panda_robot/urdf/panda.urdf'),
                                   basePosition=self._pos_init, useFixedBase=True, flags=flags)
        assert self._panda is not None, "Failed to load the panda model"

        # reset joints to home position
        num_joints = p.getNumJoints(self._panda)
        idx = 0
        for i in range(num_joints):
            joint_info = p.getJointInfo(self._panda, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                self._joint_name_to_ids[joint_name] = i

                p.resetJointState(self._panda, i, self.initial_positions[joint_name])
                p.setJointMotorControl2(self._panda, i, p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)
                idx += 1

        self.ll, self.ul, self.jr, self.rs = self.get_joint_ranges()

        if self._use_IK:
            self._home_hand_pose = [0.2, 0.0, 1.4,
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]

            self.apply_action(self._home_hand_pose)
            p.stepSimulation()
            time.sleep(self._timeStep)

        self._workspace = bullet.Sensor(self._panda,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])

        self._end_effector = bullet.get_index_by_attribute(
            self._panda, 'link_name', 'gripper_site')

        # self.add_object(change_object=change_object)
        self._format_state_query()

        #Goal Generation Process
        #self.goal_pos = self.random_goal_generation()
        self.goal_pos = np.array([-0.25, -0.75, 1.4]) #Fixed goal for demo video
        print('Printing Goal:',self.goal_pos)

        return self.get_observation()

    # def get_object_info(self):
    #     complete_object_dict, scaling = metadata.obj_path_map, metadata.path_scaling_map
    #     complete = self.object_subset is None
    #     train = (self.object_subset == 'train') or (self.object_subset == 'all')
    #     test = (self.object_subset == 'test') or (self.object_subset == 'all')
    #
    #     object_dict = {}
    #     for k in complete_object_dict.keys():
    #         in_test = (k in test_set)
    #         in_subset = (k in self.object_subset)
    #         if in_subset:
    #             object_dict[k] = complete_object_dict[k]
    #         if complete:
    #             object_dict[k] = complete_object_dict[k]
    #         if train and not in_test:
    #             object_dict[k] = complete_object_dict[k]
    #         if test and in_test:
    #             object_dict[k] = complete_object_dict[k]
    #     return object_dict, scaling

    def _set_spaces(self):
        act_dim = self.get_action_dim()
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 11
        if self.DoF > 3:
            # Add wrist theta
            observation_dim += 3

        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        self.observation_space = Dict([
            ('observation', state_space),
            ('state_observation', state_space),
            ('desired_goal', state_space),
            ('state_desired_goal', state_space),
            ('achieved_goal', state_space),
            ('state_achieved_goal', state_space),
        ])

    def _load_table(self):

        # self._panda = bullet.objects.panda_robot()
        self._table = bullet.objects.table(rgba=[.92, .85, .7, 1])
        self._base = bullet.objects.panda_base()
        # shelf
        # self._shelf1 = bullet.objects.shelf1(rgba=[.92, .85, .7, 1])
        # self._shelf2 = bullet.objects.shelf2(rgba=[.92, .85, .7, 1])
        # self._shelf3 = bullet.objects.shelf3(rgba=[1, 1, 1, 1])
        # box
        self._box = bullet.objects.box(rgba=[1, 1, 1, 1])
        self._box1 = bullet.objects.box1(rgba=[.92, .85, .7, 1])
        self._box2 = bullet.objects.box2(rgba=[.92, .85, .7, 1])
        self._box3 = bullet.objects.box3(rgba=[1,1,1,1])

        self._objectcube = bullet.objects.cube()
        self._objects = {}
        self._sensors = {}

    def sample_object_location(self):
        if self._randomize:
            return np.random.uniform(
                low=self._object_position_low, high=self._object_position_high)
        return self._fixed_object_position

    def sample_object_color(self):
        if np.random.uniform() < self.random_color_p:
            return list(np.random.choice(range(256), size=3) / 255.0) + [1]
        return None

    def sample_quat(self, object_name):
        if object_name in self.quat_dict:
            return self.quat_dict[self.curr_object]
        return deg_to_quat(np.random.randint(0, 360, size=3))

    # def _set_positions(self, pos):
    #     bullet.reset()
    #     bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
    #     self._load_table()
    #
    #     hand_pos = pos[:3]
    #     gripper = pos[self.start_obj_ind - 1]
    #     object_pos = pos[self.start_obj_ind:self.start_obj_ind + 3]
    #     object_quat = pos[self.start_obj_ind + 4:self.start_obj_ind + 7]
    #
    #     # self.add_object(change_object=False, object_position=object_pos, quat=object_quat)
    #
    #     if self.DoF > 3:
    #         hand_theta = pos[3:7]
    #     else:
    #         hand_theta = self.default_theta
    #
    #     self._format_state_query()
    #     self._prev_pos = np.array(hand_pos)
    #
    #     bullet.position_control(self._panda, self._end_effector, self._prev_pos, self.default_theta)
    #     action = np.array([0 for i in range(self.DoF)] + [gripper])
    #
    #     for _ in range(10):
    #         self.step(action)

    # def add_object(self, change_object=False, object_position=None, quat=None):
    #     # Pick object if necessary and save information
    #     if change_object:
    #         self.curr_object, self.curr_id = random.choice(list(self.object_dict.items()))
    #         self.curr_color = self.sample_object_color()
    #
    #     else:
    #         self.curr_object = self._objectcube
    #         self.curr_id = 'cube'
    #         self.curr_color = self.sample_object_color()

        # Generate random object position
        # if object_position is None:
        #     object_position = self.sample_object_location()
            # print(object_position)

        # Generate quaterion if none is given
        # if quat is None:
        #     quat = self.sample_quat(self.curr_object)

        # Spawn object above table
        # self._objects = {
        #     'obj': loader.load_shapenet_object(
        #         self.curr_id,
        #         self.scaling,
        #         object_position,
        #         quat=quat,
        #         rgba=self.curr_color)
        #     }

        # Allow the objects to land softly in low gravity
        # p.setGravity(0, 0, -1)
        # for _ in range(100):
        #     bullet.step()
        # # After landing, bring to stop
        # p.setGravity(0, 0, -10)
        # for _ in range(100):
        #     bullet.step()

    def _format_action(self, *action):
        if self.DoF == 3:
            if len(action) == 1:
                delta_pos, gripper = action[0][:-1], action[0][-1]
            elif len(action) == 2:
                delta_pos, gripper = action[0], action[1]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), gripper
        elif self.DoF == 6:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:6], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper
        elif self.DoF == 7:
            if len(action) == 1:
                delta_pos, delta_angle, gripper = action[0][:3], action[0][3:7], action[0][-1]
            elif len(action) == 3:
                delta_pos, delta_angle, gripper = action[0], action[1], action[2]
            else:
                raise RuntimeError('Unrecognized action: {}'.format(action))
            return np.array(delta_pos), np.array(delta_angle), gripper

    def get_info(self):
        # object_pos = np.asarray(self.get_object_midpoint('obj'))
        object_pos = np.asarray(bullet.get_midpoint(self._objectcube))

        height = object_pos[2]
        object_goal_distance = np.linalg.norm(object_pos - self.goal_pos)
        end_effector_pos = self.get_end_effector_pos()
        object_gripper_distance = np.linalg.norm(
            object_pos - end_effector_pos)
        gripper_goal_distance = np.linalg.norm(
            self.goal_pos - end_effector_pos)
        object_goal_success = int(object_goal_distance < self._success_threshold)
        picked_up = height > self.pickup_eps

        info = {
            'object_goal_distance': object_goal_distance,
            'object_goal_success': object_goal_success,
            'object_height': height,
            'picked_up': picked_up,
        }

        return info

    def get_contextual_diagnostics(self, paths, contexts):
        from multiworld.envs.env_util import create_stats_ordered_dict
        diagnostics = OrderedDict()
        state_key = "state_observation"
        goal_key = "state_desired_goal"
        values = []
        eps1, eps2 = [], []
        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key][self.start_obj_ind:self.start_obj_ind + 3]
            goal = contexts[i][goal_key][self.start_obj_ind:self.start_obj_ind + 3]
            distance = np.linalg.norm(state - goal)

            if self.task == 'pickup':
                values.append(state[2] > self.pickup_eps)
            if self.task == 'goal_reaching':
                values.append(distance)
                eps1.append(distance < 0.05)
                eps2.append(distance < 0.08)

        if self.task == 'pickup':
            diagnostics_key = goal_key + "/final/picked_up"
        if self.task == 'goal_reaching':
            diagnostics_key = goal_key + "/final/distance"
            diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success_0.05", eps1))
            diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success_0.08", eps2))
        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))

        values = []
        eps1, eps2 = [], []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key][self.start_obj_ind:self.start_obj_ind + 3]
                goal = contexts[i][goal_key][self.start_obj_ind:self.start_obj_ind + 3]
                distance = np.linalg.norm(state - goal)

                if self.task == 'pickup':
                    values.append(state[2] > self.pickup_eps)
                if self.task == 'goal_reaching':
                    values.append(distance)
                    eps1.append(distance < 0.05)
                    eps2.append(distance < 0.08)

        if self.task == 'pickup':
            diagnostics_key = goal_key + "/picked_up"
        if self.task == 'goal_reaching':
            diagnostics_key = goal_key + "/distance"
            diagnostics.update(create_stats_ordered_dict(goal_key + "/success_0.05", eps1))
            diagnostics.update(create_stats_ordered_dict(goal_key + "/success_0.08", eps2))

        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))
        return diagnostics


######################################## "RENDERDING" ########################################
##############################################################################################

    def render_obs(self):
        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, shadow=0, gaussian_width=5)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def render_obs_active(self):
        eef_pos_for_active_camera = self.get_end_effector_pos()
        eef_pos_for_active_camera = [float(eef_pos_for_active_camera[0]+0.15),float(eef_pos_for_active_camera[1]),float(eef_pos_for_active_camera[2])]
        eef_theta_for_active_camera = self.get_end_effector_theta()
        #print(eef_pos_for_active_camera)
        #print('Total',eef_theta_for_active_camera)
        # print('x',eef_theta_for_active_camera[0])
        # print('y',eef_theta_for_active_camera[1])
        # print('z',eef_theta_for_active_camera[2])

        view_matrix_obs_active = bullet.get_view_matrix(
            target_pos=eef_pos_for_active_camera, distance=0.35,
            yaw=eef_theta_for_active_camera[0], pitch=eef_theta_for_active_camera[1]-90, roll=eef_theta_for_active_camera[2]-270, up_axis_index=2)
        projection_matrix_obs_active = bullet.get_projection_matrix(
            self.obs_img_dim_active, self.obs_img_dim_active)

        img_active, depth, segmentation = bullet.render(
            self.obs_img_dim_active, self.obs_img_dim_active, view_matrix_obs_active,
            projection_matrix_obs_active, shadow=0, gaussian_width=0)
        if self._transpose_image:
            img_active = np.transpose(img_active, (2, 0, 1))
        return img_active

######################################## "RENDERDING" ########################################



    def set_goal(self, goal):
        self.goal_pos = goal['state_desired_goal'][self.start_obj_ind:self.start_obj_ind + 3]

    def get_image(self, width, height):
        image = np.float32(self.render_obs())
        return image

    def format_obs(self, obs):
        if len(obs.shape) == 1:
            return obs.reshape(1, -1)
        return obs

    ############################################
    ##################REWARD####################

    def get_reward(self, info):
        if self.task == 'goal_reaching':
            return info['object_goal_success'] - 1
        elif self.task == 'pickup':
            return info['picked_up'] - 1
        elif self.task == 'Pick and Place':
            return info['Goal_Success'] - 1

    def compute_reward_pu(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        height = obj_state[:, 2]
        reward = (height > self.pickup_eps) - 1
        return reward

    def compute_reward_gr(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        object_goal_success = object_goal_distance < self._success_threshold
        return object_goal_success - 1


    def compute_reward(self, obs, actions, next_obs, contexts):
        obj_state = self.format_obs(next_obs['state_observation'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        obj_goal = self.format_obs(contexts['state_desired_goal'])[:, self.start_obj_ind:self.start_obj_ind + 3]
        object_goal_distance = np.linalg.norm(obj_state - obj_goal, axis=1)
        Goal_Success = object_goal_distance < self._success_threshold
        return Goal_Success - 1

    def compute_reward(self, obs, actions, next_obs, contexts):
        if self.task == 'goal_reaching':
            return self.compute_reward_gr(obs, actions, next_obs, contexts)
        elif self.task == 'pickup':
            return self.compute_reward_pu(obs, actions, next_obs, contexts)
        elif self.task == 'Pick and Place':
            return self.compute_reward(obs, actions, next_obs, contexts)

    ##################REWARD####################
    ############################################

    def get_object_deg(self):
        # object_info = bullet.get_body_info(self._objects['obj'],
        #                                    quat_to_deg=True)
        object_info = bullet.get_body_info(self._objectcube,
                                           quat_to_deg=True)
        return object_info['theta']

    def get_hand_deg(self):
        return bullet.get_link_state(self._panda, self._end_effector,
            'theta', quat_to_deg=True)

    def get_observation(self):
        left_tip_pos = bullet.get_link_state(
            self._panda, 'panda_finger_joint1', keys='pos')
        right_tip_pos = bullet.get_link_state(
            self._panda, 'panda_finger_joint2', keys='pos')
        left_tip_pos = np.asarray(left_tip_pos)
        right_tip_pos = np.asarray(right_tip_pos)
        hand_theta = bullet.get_link_state(self._panda, self._end_effector,
            'theta', quat_to_deg=False)
        #print('obs_ hand theta',hand_theta)
        gripper_tips_distance = [np.linalg.norm(
            left_tip_pos - right_tip_pos)]
        end_effector_pos = self.get_end_effector_pos()

        #Spawning random objects code
        # object_info = bullet.get_body_info(self._objects['obj'],
        #                                    quat_to_deg=False)

        # Cube code
        object_info = bullet.get_body_info(self._objectcube, quat_to_deg=False)
        #print('pos cube',object_info)
        object_pos = object_info['pos']
        object_theta = object_info['theta']

        if self.DoF > 3:
            observation = np.concatenate((
                end_effector_pos, hand_theta, gripper_tips_distance,
                object_pos, object_theta))
            goal_pos = np.concatenate((
                self.goal_pos, hand_theta, gripper_tips_distance,
                self.goal_pos, object_theta))
        else:
            observation = np.concatenate((
                end_effector_pos, gripper_tips_distance,
                object_pos, object_theta))
            goal_pos = np.concatenate((
                end_effector_pos, gripper_tips_distance,
                self.goal_pos, object_theta))

        obs_dict = dict(
            observation=observation,
            state_observation=observation,
            desired_goal=goal_pos,
            state_desired_goal=goal_pos,
            achieved_goal=observation,
            state_achieved_goal=observation,
            )

        return obs_dict

#################################################################################################
#################################################################################################
#################################################################################################

    def step(self, *action):

        pos = bullet.get_link_state(self._panda, self._end_effector, 'pos')
        theta = [m.pi, 0, 0]
        #theta = p.getLinkState(self._panda, self.end_eff_idx)[5]
        #print('theta',theta)

        #print('action_before format', action)
        delta_pos, delta_angle, gripper = self._format_action(*action)

        #print('gripper',gripper)
        if gripper == -1:
            self.pre_grasp()
            p.stepSimulation()
        elif gripper == 1:
            self.grasp(self._objectcube)
            p.stepSimulation()

        adjustment = 0.1
        pos += delta_pos * adjustment

        # pos = np.clip(pos, self._pos_low, self._pos_high)
        #print(delta_angle)

        theta_adjustment = 0.2
        theta += delta_angle*theta_adjustment
        #print('delta_action', pos, theta)

        pos_and_theta = np.append(pos,theta)
        #print('action_to_apply', pos_and_theta)

        self.apply_action(pos_and_theta)
        p.stepSimulation()
        #print('eef_pos with action', pos)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False

        return observation, reward, done, info

    def demo_reset(self):
        self.grip = -1.
        self.done = False
        reset_obs = self.reset()
        return reset_obs

    def get_demo_action(self):
        action, done = self.move_obj(self.goal_pos)
        self.done = done or self.done
        action = np.append(action, [self.grip])
        #action = np.append(action, [self.action_theta])
        #action = np.random.normal(action, 0.1)
        action = np.clip(action, a_min=-3.14, a_max=3.14)
        return action

    def move_obj(self, goal):
        ee_pos = self.get_end_effector_pos()
        target_pos = np.array(bullet.get_body_info(self._objectcube)['pos'])
        # target_pos = np.array(bullet.get_body_info(self._objects['obj'], quat_to_deg=False)['pos']) + adjustment
        print('cube', target_pos)
        #print('eef', ee_pos)

        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.02
        on_top = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.01
        done = (np.linalg.norm(target_pos - goal) < 0.025) or self.done

        on_drop_height = 0.1 < target_pos[2] - goal[2] < 0.15
        placing = np.linalg.norm(goal[:2] - target_pos[:2]) < 0.3
        placing_near = np.linalg.norm(goal[:2] - target_pos[:2]) < 0.03

        # check = self.check_contact_fingertips(self._objectcube)
        # print('check', check)
        grasp = int(self.check_contact_fingertips(self._objectcube)[0]) == 2
        #print('Check Grasp',grasp)

        #print(self.shelf_chosen)
        ori_angle = np.array([0, 0, 0])
        change_angle = np.array([0, m.pi, 0])

        if done and self.goal_near == 1:
            print('Finished')
            action = np.array([0., 0, 0.05])
            action = np.append(action, ori_angle)
            self.grip = -1

        if not grasp and  self.goal_near == 0:
            if not aligned and not on_top:
                print('Stage 1: Appraoching')
                action = target_pos - ee_pos
                action *= 2
                action = np.append(action, ori_angle)
                self.grip = -1

            elif aligned and not on_top:
                print('Stage 2: Not on top')
                action = np.array([0., 0., -0.5])
                action = np.append(action, ori_angle)
                self.grip = -1

            elif not aligned and on_top:
                print('Stage 2: Aligning')
                action = target_pos - ee_pos
                action[2] = 0
                action *= 0.8
                action = np.append(action, ori_angle)
                self.grip = -1

            elif aligned and on_top:
                print('Stage 4: Grasping')
                action = np.array([0., 0., 0])
                action = np.append(action, ori_angle)
                self.grip = 1

        if grasp:
            if not on_drop_height and not placing:
                print('Stage 5: Going to goal1')
                action = np.array([0., 0., 0.3])
                action = np.append(action, ori_angle)
                self.grip = 1

            if on_drop_height and not placing:
                print('Stage 6: Going to goal2')
                action = goal - target_pos
                action[2] = 0
                action *= 0.5
                action = np.append(action, ori_angle)
                self.grip = 1

            if placing and not placing_near:
                print('Stage 8: Alinging to Goal')
                action = goal - target_pos
                action[2] = 0
                action *= 1.2
                action = np.append(action, ori_angle)
                self.grip = 1

            if placing_near and not done:
                self.goal_near = 1
                print('Stage 9: Dropping to Goal')
                action = np.array([0., 0., -0.1])
                action = np.append(action, ori_angle)
                self.grip = 1

        return action, done


