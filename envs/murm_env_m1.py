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
from math import pi

class MURMENV_m1(PandaBaseEnv):

    def __init__(self,
                 goal_pos=(0, 0, 0),
                 reward_type='shaped',
                 reward_min=-2.5,
                 randomize=True,
                 observation_mode='state',
                 #todo: Image Dimension
                 obs_img_dim=64,
                 obs_img_dim_active=64,
                 success_threshold=0.03,
                 transpose_image=False,
                 invisible_robot=False,
                 object_subset='all',
                 use_bounding_box=True,
                 random_color_p=1,
                 test_env=False,
                 env_type=None,
                 DoF = 3,
                 *args,
                 **kwargs
                 ):
        is_set = object_subset in ['test', 'train', 'all']
        is_list = type(object_subset) == list
        assert is_set or is_list

        self.goal_pos = np.asarray(goal_pos)
        self._reward_type = reward_type
        self._reward_min = reward_min
        self._randomize = randomize
        self._observation_mode = observation_mode
        self._transpose_image = transpose_image
        self._invisible_robot = invisible_robot
        self.image_shape = (obs_img_dim, obs_img_dim)
        self.image_length = np.prod(self.image_shape) * 3  # image has 3 channels
        self.random_color_p = random_color_p
        self.use_bounding_box = use_bounding_box
        self.object_subset = object_subset
        self.test_env = test_env
        self.DoF = DoF
        self.obj_index = 0

        # _obj POSITION wall
        self._object_position_low = (0.35, -0.1, 1.017)
        self._object_position_high = (0.45, 0.1, 1.017)
        self._fixed_object_position = np.array([0.45, 0, 1.017])

        self._fixed_object_position1 = np.array([0.45, 0, 1.048])
        self._object_position_low1 = (0.35, -0.1, 1.048)
        self._object_position_high1 = (0.45, 0.1, 1.048)

        # # _obj POSITION
        # self._object_position_low = (0.4, -0.1, 1.03)
        # self._object_position_high = (0.5, 0.1, 1.03)
        # self._fixed_object_position = np.array([0.45, 0, 1.03])
        #
        # self._fixed_object_position1 = np.array([0.45, 0, 1.048])
        # self._object_position_low1 = (0.4, -0.1, 1.048)
        # self._object_position_high1 = (0.5, 0.1, 1.048)

        self.default_theta = bullet.deg_to_quat([180, 0, 0])
        self._success_threshold = success_threshold

        #Global Camera
        self.obs_img_dim = obs_img_dim #+.15
        #Active Camera
        self.obs_img_dim_active = obs_img_dim_active
        self.dt = 0.1

        super().__init__(*args, **kwargs)
        self._max_force = 100
        self._timeStep = 1. / 240.

        self._view_matrix_obs = bullet.get_view_matrix(
            target_pos=[0.7, -0.2, 1.3], distance=0.4, # [0.8, 0, 1.5], distance=0.8,
            yaw=90, pitch=-20, roll=0, up_axis_index=2)
        self._projection_matrix_obs = bullet.get_projection_matrix(
            self.obs_img_dim, self.obs_img_dim)

    def random_goal_generation(self):
        boxesgoals = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        chosen_box = random.choice(boxesgoals)

        if chosen_box == 1:
            goal = np.array([-0.05, -0.4, 1.037])
        elif chosen_box == 2:
            goal = np.array([0.1, -0.4, 1.037])
        elif chosen_box == 3:
            goal = np.array([0.25, -0.4, 1.037])

        elif chosen_box == 4:
            goal = np.array([-0.05, -0.55, 1.037])
        elif chosen_box == 5:
            goal = np.array([0.1, -0.55, 1.037])
        elif chosen_box == 6:
            goal = np.array([0.25, -0.55, 1.037])

        elif chosen_box == 7:
            goal = np.array([-0.05, -0.7, 1.037])
        elif chosen_box == 8:
            goal = np.array([0.1, -0.7, 1.037])
        elif chosen_box == 9:
            goal = np.array([0.25, -0.7, 1.037])

        return goal

    def reset(self, change_object=False):

        # Load Environment
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
            self._home_hand_pose = [0.45, 0.0, 1.4,
                                    min(m.pi, max(-m.pi, m.pi)),
                                    min(m.pi, max(-m.pi, 0)),
                                    min(m.pi, max(-m.pi, 0))]

            self.apply_action(self._home_hand_pose)
            p.stepSimulation()
            time.sleep(self._timeStep)

        self._end_effector = bullet.get_index_by_attribute(
            self._panda, 'link_name', 'gripper_site')

        # Random Color and Shape of Obj
        self._obj = self.random_obj_generation()
        # print('cube 0 / prism1 1 / prism2 2 / ', self.obj_index)
        rgba = self.sample_object_color()
        for i in range(-1, 5):
            rgba = rgba
            p.changeVisualShape(self._obj, i, rgbaColor=rgba)
        self._format_state_query()

        #Goal Generation Process
        self.goal_pos = self.random_goal_generation()
        #self.goal_pos = np.array([0.25, -0.7, 1.05]) #Fixed goal for demo video
        print('Printing Goal:', self.goal_pos)

        return self.get_observation()

    def sample_object_location(self):
        if self.obj_index == 0 or self.obj_index == 1:
            initial_random_pos = np.random.uniform(low=self._object_position_low, high=self._object_position_high)
            print('Initial pos', initial_random_pos)
        elif self.obj_index == 2:
            initial_random_pos = np.random.uniform(low=self._object_position_low1, high=self._object_position_high1)
            print('obj2 Initial pos', initial_random_pos)
        else:
            print('No Obj')
        return initial_random_pos

    def sample_object_color(self):
        a = list(np.random.choice(range(256), size=3) / 255.0) + [1]
        #print('color', a)
        return a

    def random_obj_generation(self):
        random_shape = ['cube', 'rectangularprism', 'tetris1', 'tetris2']
        # chosen_shape = random.choice(random_shape)

        chosen_shape = 'cube'
        # chosen_shape = 'rectangularprism'
        # chosen_shape = 'tetris1'
        # chosen_shape = 'tetris2'

        # chosen_shape = 'rectangularprism2' #Bottle
        if chosen_shape == 'cube':
            self.obj_index = 0
            obj = bullet.objects.cube(pos=self.sample_object_location())

        elif chosen_shape == 'rectangularprism':
            self.obj_index = 1
            obj = bullet.objects.rectangularprism(pos=self.sample_object_location())  # [0.5, 0.1, 1.03])

        elif chosen_shape == 'tetris1':
            self.obj_index = 1
            obj = bullet.objects.tetris1(pos=self.sample_object_location())

        elif chosen_shape == 'tetris2':
            self.obj_index = 1
            obj = bullet.objects.tetris2(pos=self.sample_object_location())

        elif chosen_shape == 'rectangularprism2':
            self.obj_index = 2
            obj = bullet.objects.rectangularprism2(pos=self.sample_object_location())

        else:
            exit()

        return obj

    def run_for_goal(self):
        image_check_save_path = "/media/jang/jang/0ubuntu/image_dataset/Images_produced_for_goals/Wall/"
        a, q = p.getBasePositionAndOrientation(self._obj)
        p.resetBasePositionAndOrientation(self._obj, self.goal_pos, q)

        self.goal_positions = {
            'panda_joint1': -1.94, 'panda_joint2': 0.427, 'panda_joint3': 0.153,
            'panda_joint4': -2.3415, 'panda_joint5': -0.1715, 'panda_joint6': 2.7593,
            'panda_joint7': -0.8575, 'panda_finger_joint1': 0.02, 'panda_finger_joint2': 0.02,
        }

        num_joints = p.getNumJoints(self._panda)
        # print('joints',num_joints)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self._panda, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                p.resetJointState(self._panda, i, self.goal_positions[joint_name])
                p.setJointMotorControl2(self._panda, i, p.POSITION_CONTROL,
                                        targetPosition=self.goal_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)

        target_pos_check = np.array(bullet.get_body_info(self._obj)['pos'])
        ee_pos_check = self.get_end_effector_pos()
        # print('obj, ee pos for goal', target_pos_check, ee_pos_check)

        goal_global = np.uint8(self.render_obs())
        goal_active = np.uint8(self.render_obs_active())

        num_joints = p.getNumJoints(self._panda)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self._panda, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                p.resetJointState(self._panda, i, self.initial_positions[joint_name])
                p.setJointMotorControl2(self._panda, i, p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)

        p.resetBasePositionAndOrientation(self._obj, a, q)

        reset_global = np.uint8(self.render_obs())
        reset_active = np.uint8(self.render_obs_active())

        np.save(image_check_save_path+"m2_1.npy", goal_global)
        np.save(image_check_save_path+"m2_2.npy", goal_active)
        np.save(image_check_save_path+"m2_3.npy", reset_global)
        np.save(image_check_save_path+"m2_4.npy", reset_active)

        return goal_global, goal_active

    def run_for_goal2(self):
        image_check_save_path = "/media/jang/jang/0ubuntu/image_dataset/Images_produced_for_goals/"
        a, q = p.getBasePositionAndOrientation(self._obj)
        p.resetBasePositionAndOrientation(self._obj, self.goal_pos, q)

        # self.goal_positions = {
        #     'panda_joint1': -1.94, 'panda_joint2': 0.427, 'panda_joint3': 0.153,
        #     'panda_joint4': -2.3415, 'panda_joint5': -0.1715, 'panda_joint6': 2.7593,
        #     'panda_joint7': -0.8575, 'panda_finger_joint1': 0.02, 'panda_finger_joint2': 0.02,
        # }
        quaternion = p.getQuaternionFromEuler([m.pi, 0, 0])
        # print('quaternion', quaternion)
        final_pos = self.goal_pos + np.array([-0.1, 0, 0.075])
        IK = p.calculateInverseKinematics(self._panda, 11, final_pos, targetOrientation=quaternion, maxNumIterations=500, residualThreshold=0.001)
        # for i in range(15):
        #     print('link', p.getLinkState(self._panda, i))

        # print('IK', IK)
        self.goal_positions = {
             'panda_joint1': IK[0], 'panda_joint2': IK[1], 'panda_joint3': IK[2],
             'panda_joint4': IK[3], 'panda_joint5': IK[4], 'panda_joint6': IK[5],
             'panda_joint7': IK[6], 'panda_finger_joint1': 0.02, 'panda_finger_joint2': 0.02,}

        num_joints = p.getNumJoints(self._panda)
        # print('joints',num_joints)

        for i in range(num_joints):
            joint_info = p.getJointInfo(self._panda, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                p.resetJointState(self._panda, i, self.goal_positions[joint_name])
                p.setJointMotorControl2(self._panda, i, p.POSITION_CONTROL,
                                        targetPosition=self.goal_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)

        target_pos_check = np.array(bullet.get_body_info(self._obj)['pos'])
        ee_pos_check = self.get_end_effector_pos()
        # print('obj, ee pos for goal', target_pos_check, ee_pos_check)

        goal_global = np.uint8(self.render_obs())
        goal_active = np.uint8(self.render_obs_active())

        num_joints = p.getNumJoints(self._panda)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self._panda, i)
            joint_name = joint_info[1].decode("UTF-8")
            joint_type = joint_info[2]

            if joint_type is p.JOINT_REVOLUTE or joint_type is p.JOINT_PRISMATIC:
                assert joint_name in self.initial_positions.keys()

                p.resetJointState(self._panda, i, self.initial_positions[joint_name])
                p.setJointMotorControl2(self._panda, i, p.POSITION_CONTROL,
                                        targetPosition=self.initial_positions[joint_name],
                                        positionGain=0.2, velocityGain=1.0)

        p.resetBasePositionAndOrientation(self._obj, a, q)

        reset_global = np.uint8(self.render_obs())
        reset_active = np.uint8(self.render_obs_active())

        np.save(image_check_save_path+"v2_1.npy", goal_global)
        np.save(image_check_save_path+"v2_2.npy", goal_active)
        np.save(image_check_save_path+"v2_3.npy", reset_global)
        np.save(image_check_save_path+"v2_4.npy", reset_active)

        return goal_global, goal_active

    def _set_spaces(self):
        act_dim = self.get_action_dim()
        act_bound = 1
        act_high = np.ones(act_dim) * act_bound
        self.action_space = gym.spaces.Box(-act_high, act_high)

        observation_dim = 3
        observation_dim1 = 4
        obs_bound = 100
        obs_high = np.ones(observation_dim) * obs_bound
        state_space = gym.spaces.Box(-obs_high, obs_high)

        obs_high1 = np.ones(observation_dim1) * obs_bound
        state_space1 = gym.spaces.Box(-obs_high1, obs_high1)

        self.observation_space = Dict([
            ('state_observation', state_space),
            ('robot_state_observation', state_space1),
            ('state_desired_goal', state_space),
        ])

    def _load_table(self):
        self._table = bullet.objects.table(rgba=[1, 1, 1, 1])
        self._base = bullet.objects.panda_base()
        self._box1 = bullet.objects.box1()
        self._box2 = bullet.objects.box2()
        self._box3 = bullet.objects.box3()
        self._box4 = bullet.objects.box4()
        self._box5 = bullet.objects.box5()
        self._box6 = bullet.objects.box6()
        self._box7 = bullet.objects.box7()
        self._box8 = bullet.objects.box8()
        self._box9 = bullet.objects.box9()

        self._wall = bullet.objects.wall()
        #self._wall2 = bullet.objects.wall2()
        # Wall color Thick Brown "0.55 0.35 0.17 1"
        self._objects = {}
        self._sensors = {}

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

    def get_contextual_diagnostics(self, paths, contexts):
        from multiworld.multiworld.envs.env_util import create_stats_ordered_dict

        diagnostics = OrderedDict()
        state_key = "state_observation" # obj pos
        goal_key = "state_desired_goal"
        values = []
        eps1, eps2 = [], []

        # print('check paths in diagnostics', paths[0])
        # print('check contexts in diagnostics', contexts)
        # print('length of paths = 5 ?', len(paths))

        for i in range(len(paths)):
            state = paths[i]["observations"][-1][state_key]
            print('Final obj State', state)

            goal = paths[i]["observations"][-1][goal_key]
            print('Given Goal State', goal)

            distance = np.linalg.norm(state - goal)

            values.append(distance)
            eps1.append(distance < 0.03)
            eps2.append(distance < 0.1)

        diagnostics_key = goal_key + "/final/distance"
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success", eps1))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/final/success_close", eps2))
        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))

        values = []
        eps1, eps2, eps3 = [], [], []
        for i in range(len(paths)):
            for j in range(len(paths[i]["observations"])):
                state = paths[i]["observations"][j][state_key]
                # print('state each', state)
                state_z_pos = paths[i]["observations"][j][state_key][2]
                # print('checking z coordinates', state_z_pos)
                initial_z_pos = np.array(1.03)
                height = np.linalg.norm(state_z_pos-initial_z_pos)

                goal = paths[i]["observations"][-1][goal_key]
                # print('goal state each', goal)

                distance = np.linalg.norm(state - goal)
                values.append(distance)
                eps1.append(distance < 0.03)
                eps2.append(distance < 0.1)
                eps3.append(height > 0.1)

        diagnostics_key = goal_key + "/distance"
        diagnostics.update(create_stats_ordered_dict(goal_key + "/success", eps1))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/success_close", eps2))
        diagnostics.update(create_stats_ordered_dict(goal_key + "/checking_whether_picked_up", eps3))

        diagnostics.update(create_stats_ordered_dict(diagnostics_key, values))
        return diagnostics

######################################## "RENDERDING" ########################################
##############################################################################################

    def render_obs(self):

        # img_tuple1 = p.getCameraImage(width=self.obs_img_dim,
        #                  height=self.obs_img_dim,
        #                  viewMatrix=self._view_matrix_obs,
        #                  projectionMatrix=self._projection_matrix_obs,
        #                  shadow=1,
        #                  lightDirection=[1,1,1],
        #                  renderer=p.ER_TINY_RENDERER)
        # _, _, img1, depth, segmentation = img_tuple1
        #
        # img = img1[:, :, :-1]

        img, depth, segmentation = bullet.render(
            self.obs_img_dim, self.obs_img_dim, self._view_matrix_obs,
            self._projection_matrix_obs, lightdistance=0.1, shadow=0, light_direction=[1, 1, 1], gaussian_width=5)
        if self._transpose_image:
            img = np.transpose(img, (2, 0, 1))
        return img

    def render_obs_active(self):
        eef_pos_for_active_camera = self.get_end_effector_pos()
        eef_pos_for_active_camera = [float(eef_pos_for_active_camera[0]+0.085),float(eef_pos_for_active_camera[1]),float(eef_pos_for_active_camera[2])]
        eef_theta_for_active_camera = self.get_end_effector_theta()
        #print(eef_pos_for_active_camera)
        #print('Total',eef_theta_for_active_camera)
        # print('x',eef_theta_for_active_camera[0])
        # print('y',eef_theta_for_active_camera[1])
        # print('z',eef_theta_for_active_camera[2])

        view_matrix_obs_active = bullet.get_view_matrix(
            target_pos=eef_pos_for_active_camera, distance=0.2,
            yaw=eef_theta_for_active_camera[0], pitch=eef_theta_for_active_camera[1]-90, roll=eef_theta_for_active_camera[2]-270, up_axis_index=2)
        projection_matrix_obs_active = bullet.get_projection_matrix(
            self.obs_img_dim_active, self.obs_img_dim_active)

        # img_tuple2 = p.getCameraImage(width=self.obs_img_dim_active,
        #                  height=self.obs_img_dim_active,
        #                  viewMatrix=view_matrix_obs_active,
        #                  projectionMatrix=projection_matrix_obs_active,
        #                  shadow=1,
        #                  lightDirection=[1,1,1],
        #                  renderer=p.ER_TINY_RENDERER)
        # _, _, img2, depth2, segmentation2 = img_tuple2
        #
        # img_active = img2[:, :, :-1]

        img_active, depth, segmentation = bullet.render(
            self.obs_img_dim_active, self.obs_img_dim_active, view_matrix_obs_active,
            projection_matrix_obs_active, lightdistance=0.1, shadow=0, light_direction=[1, 1, 1], gaussian_width=5)
        if self._transpose_image:
            img_active = np.transpose(img_active, (2, 0, 1))
        return img_active

    def get_image(self, width, height, camera):
        if camera == 'global':
            image = np.float32(self.render_obs())
        elif camera == 'active':
            image = np.float32(self.render_obs_active())
        return image

    # def set_goal(self, goal):
    #     self.goal_pos = goal['state_desired_goal'][self.start_obj_ind:self.start_obj_ind + 3]
    #
    # def format_obs(self, obs):
    #     if len(obs.shape) == 1:
    #         return obs.reshape(1, -1)
    #     return obs

    ############################################
    ##################REWARD####################

    def get_info(self):
        object_pos = np.asarray(bullet.get_body_info(self._obj)['pos'])
        object_goal_distance = np.linalg.norm(object_pos - self.goal_pos)
        object_goal_success = int(object_goal_distance < self._success_threshold)
        info = {'Goal_success': object_goal_success}
        return info

    def get_reward(self, info):
        return info['Goal_success'] - 1

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

        object_info = bullet.get_body_info(self._obj, quat_to_deg=False)
        object_pos = object_info['pos']
        #object_theta = object_info['theta']

        observation = np.concatenate((
            end_effector_pos, gripper_tips_distance))

        obj_observation = np.asarray(object_pos)

        goal_pos = np.asarray(self.goal_pos)
        # print('DOF  @@  HERE')

        obs_dict = dict(
            state_observation=obj_observation,
            robot_state_observation=observation,
            state_desired_goal=goal_pos,
        )

        return obs_dict

    def step(self, *action):
        # Joint Initialization Code
        # a0 = bullet.get_joint_state(self._panda, 0)
        # a1 = bullet.get_joint_state(self._panda, 1)
        # a2 = bullet.get_joint_state(self._panda, 2)
        # a3 = bullet.get_joint_state(self._panda, 3)
        # a4 = bullet.get_joint_state(self._panda, 4)
        # a5 = bullet.get_joint_state(self._panda, 5)
        # a6 = bullet.get_joint_state(self._panda, 6)
        # print('jointinfo0', a0, a1, a2, a3, a4, a5, a6)

        pos = bullet.get_link_state(self._panda, self._end_effector, 'pos')
        #theta = p.getLinkState(self._panda, self.end_eff_idx)[5][:3]
        # a = theta[0]*pi
        # b = theta[1]*pi
        # c = theta[2]*pi
        # theta = np.array([a, b, c])
        #print('theta',theta)
        theta = [m.pi, 0, 0] # Theta Fixed to pi

        #print('action_before format', action)
        # delta_pos, delta_angle, gripper = self._format_action(*action)
        delta_pos, gripper = self._format_action(*action)

        #print('gripper',gripper)
        if gripper == -1:
            self.pre_grasp()
            p.stepSimulation()
        elif gripper == 1 and self.obj_index == 2:
            self.grasp2(self._obj)
            p.stepSimulation()
        elif gripper == 1 and self.obj_index == 1:
            self.grasp1(self._obj)
            p.stepSimulation()
        elif gripper == 1 and self.obj_index == 0:
            self.grasp(self._obj)
            p.stepSimulation()

        adjustment = 0.1
        pos += delta_pos * adjustment

        # pos = np.clip(pos, self._pos_low, self._pos_high)
        #print(delta_angle)

        # theta_adjustment = 0.2
        # theta += delta_angle*theta_adjustment
        #print('delta_action', pos, theta)

        pos_and_theta = np.append(pos, theta)
        #print('action_to_apply', pos_and_theta)

        self.apply_action(pos_and_theta)

        p.stepSimulation()
        #print('eef_pos with action', pos)

        # Get tuple information
        observation = self.get_observation()
        info = self.get_info()
        reward = self.get_reward(info)
        done = False
        # self.timeStep += 1

        return observation, reward, done, info

    def demo_reset(self):
        self.grip = -1.
        self.done = False
        self.trigger = 0
        self.timeStep = 0
        self.taketime = 0
        self.x = 0
        self.y = 0
        self.z = 0.5
        self.xx = 0.5
        self.up = 0
        self.goal_near = 0
        self.achieve_check = 0
        self.time_add = 0
        reset_obs = self.reset()
        return reset_obs

    def get_demo_action(self):
        action, done = self.my_action(self.goal_pos)
        self.done = done or self.done
        action = np.append(action, [self.grip])
        #action = np.append(action, [self.action_theta])
        #action = np.random.normal(action, 0.1)
        action = np.clip(action, a_min=-3.14, a_max=3.14)
        return action

    def my_action(self, goal):
        ee_pos = self.get_end_effector_pos()
        target_pos = np.array(bullet.get_body_info(self._obj)['pos'])
        adjustment = np.array([0, 0, 0.014])
        adjustment1 = np.array([0, 0, 0.1])
        adjustment2 = np.array([0, 0, 0.01])
        if self.obj_index == 2:
            target_pos = np.array(bullet.get_body_info(self._obj)['pos']) + adjustment
            goal = goal + adjustment2

        target_pos2 = np.array(bullet.get_body_info(self._obj)['pos']) + adjustment1

        ee_set_pos = np.array([0.1, -0.55, 1.5])
        checking_pos = self.goal_pos+np.array([-0.1, 0, 0.075])
        #print('cube', target_pos)
        #print('eef', ee_pos)

        aligned = np.linalg.norm(target_pos[:2] - ee_pos[:2]) < 0.005
        done = (np.linalg.norm(target_pos - goal) < 0.03) or self.done
        on_top = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.015
        if self.obj_index == 1 or self.obj_index == 0:
            on_top = np.linalg.norm(target_pos[2] - ee_pos[2]) < 0.01
        on_drop_height = 0.095 < target_pos[2] - goal[2] < 0.125
        placing = np.linalg.norm(ee_set_pos[:2] - target_pos[:2]) < 0.05
        placing_near = np.linalg.norm(goal[:2] - target_pos[:2]) < 0.005

        # check = self.check_contact_fingertips(self._obj)
        # print('check', check)
        grasp = int(self.check_contact_fingertips(self._obj)[0]) == 2
        #print('Check Grasp',grasp)

        #print(self.shelf_chosen)
        o_angle = ([pi, 0, 0])
        curr_angle = np.array(p.getLinkState(self._panda, self.end_eff_idx)[5][:3])
        ori_angle = ([0, 0, 0]) #= (o_angle - curr_angle)*5
        turn_angle = np.array([ori_angle[0]/2, ori_angle[1]/2, -pi/2*3])
        turned = p.getLinkState(self._panda, self.end_eff_idx)[5][2] > pi*1.4
        #print('theta', turned)
        #print('trigger',self.trigger)
        #
        # if self.timeStep > 200:
        #     print('time', self.timeStep)

        action = np.array([0, 0, 0])
        self.grip = 1

        if done and self.goal_near == 1 and self.achieve_check == 0:
            #print('Finished')
            action = np.array([0, 0, 0])
            # action = np.append(action, ori_angle)
            self.grip = -1
            self.time_add += 1

        if done and 12 > self.time_add > 7:
            action = np.array([0, 0, 0.1])
            self.grip = -1
            self.achieve_check += 0.21

        if done and self.achieve_check > 1:
            action = checking_pos - ee_pos
            action *= self.xx * 1.5
            self.grip = -1
            self.xx += 0.3

        if not grasp and self.goal_near == 0:
            if not aligned and not on_top:
                #print('Stage 1: Approaching')
                action = target_pos2 - ee_pos
                self.z += 0.1
                #print('z', self.z)
                action *= self.z*1.5
                # action = np.append(action, ori_angle)
                self.grip = -1

            elif aligned and not on_top:
                #print('Stage 2: Not on top')
                action = np.array([0., 0., -0.3])
                # action = np.append(action, ori_angle)
                self.grip = -1

            elif not aligned and on_top:
                #print('Stage 2: Aligning')
                action = target_pos - ee_pos
                action[2] = 0
                action *= 0.5
                # action = np.append(action, ori_angle)
                self.grip = -1

            elif aligned and on_top:
                #print('Stage 4: Grasping')
                action = np.array([0., 0., 0])
                # action = np.append(action, ori_angle)
                self.grip = 1

        if grasp and self.taketime < 6:
            action = np.array([0., 0., 0])
            self.grip = 1
            self.taketime += 1

        if grasp and self.taketime > 5:
            if not on_drop_height and not placing and not self.trigger: # and not turned:
                #print('Stage 5: Going to Placing(up)')
                action = np.array([0., 0., 0.2])
                self.up += 0.1
                action *= self.up*1
                # action = np.append(action, ori_angle)
                self.grip = 1

            if on_drop_height and not placing and not self.trigger: # and not turned:
                #print('Stage 6: Going to Placing 2')
                action = ee_set_pos - target_pos
                action[2] = 0
                self.x += 0.1
                #print('x', self.x)
                action *= self.x*0.6
                # action = np.append(action, ori_angle)
                self.grip = 1

            # if on_drop_height and placing and not turned:
            #     #print('Stage 7: Turning')
            #     action = np.array([0., 0., 0.])
            #     action = np.append(action, ori_angle)
            #     self.grip = 1
            if placing and on_drop_height:
                self.trigger = 1

            if self.trigger == 1 and not placing_near:
                #print('Stage 8: Alinging to Goal')
                action = goal - target_pos
                action[2] = 0
                self.y += 0.1
                action *= self.y * 1
                # action = np.append(action, ori_angle)
                self.grip = 1

            if placing_near and not done and self.trigger == 1:
                self.goal_near = 1
                #print('Stage 9: Dropping to Goal')
                action = np.array([0., 0., -0.2])
                # action = np.append(action, ori_angle)
                self.grip = 1

        return action, done
