import numpy as np
import gym
import pdb
import math as m
import pybullet as p

import roboverse.bullet as bullet
from roboverse.envs.serializable import Serializable


class PandaBaseEnv_t3(gym.Env, Serializable):

    initial_positions = {
        'panda_joint1': 0.01980299680137533, 'panda_joint2': -0.500723153138898, 'panda_joint3': -0.016822874075914835,
        'panda_joint4': -2.3816251141454, 'panda_joint5': -0.008481703547671847, 'panda_joint6': 1.881267067869108,
        'panda_joint7': 0.7930317125684555, 'panda_finger_joint1': 0.04, 'panda_finger_joint2': 0.04,
    }

    def __init__(self,
                 img_dim=64,
                 gui=False,
                 action_scale=.2,
                 action_repeat=10,
                 timestep=1./240, #1./240 (1./360 is more realistic, but harder)
                 solver_iterations=150,
                 gripper_bounds=[-1, 1],
                 pos_init=[0, 0.0, 1.0],

                 # task3
                 pos_low=[0.25, -0.15, 1.0],
                 pos_high=[0.55, 0.25, 1.25],

                 max_force=1000.,
                 visualize=True,
                 use_IK=1,
                 control_orientation=0,
                 control_eu_or_quat=0,
                 joint_action_space=9
                 ):

        self._use_IK = use_IK
        self._control_orientation = control_orientation
        self.joint_action_space = joint_action_space
        self._control_eu_or_quat = control_eu_or_quat
        self._eu_lim = [[-m.pi, m.pi], [-m.pi, m.pi], [-m.pi, m.pi]]
        self.end_eff_idx = 11  # 8
        self._home_hand_pose = []
        self._num_dof = 7
        self._joint_name_to_ids = {}

        self._gui = gui
        self._action_scale = action_scale
        self._action_repeat = action_repeat
        self._timestep = timestep
        self._solver_iterations = solver_iterations
        self._gripper_bounds = gripper_bounds
        self._pos_init = pos_init
        self._pos_low = pos_low
        self._pos_high = pos_high
        self._max_force = max_force
        self._visualize = visualize
        self._id = 'PandaBaseEnv'

        self.theta = bullet.deg_to_quat([180, 0, 0])

        bullet.connect_headless(self._gui)
        # self.set_reset_hook()
        self._set_spaces()

        self._img_dim = img_dim
        self._view_matrix = bullet.get_view_matrix()
        self._projection_matrix = bullet.get_projection_matrix(self._img_dim, self._img_dim)


    def get_joint_ranges(self):
        lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []
        for joint_name in self._joint_name_to_ids.keys():
            jointInfo2 = p.getJointInfo(self._panda, self._joint_name_to_ids[joint_name])

            ll, ul = jointInfo2[8:10]
            jr = ul - ll
            # For simplicity, assume resting state == initial state
            rp = self.initial_positions[joint_name]
            lower_limits.append(ll)
            upper_limits.append(ul)
            joint_ranges.append(jr)
            rest_poses.append(rp)

        return lower_limits, upper_limits, joint_ranges, rest_poses

    def apply_action(self, action, max_vel=-1):

        if self._use_IK:
            # ------------------ #
            # --- IK control --- #
            # ------------------ #

            if not (len(action) == 3 or len(action) == 6 or len(action) == 7):
                raise AssertionError('number of action commands must be \n- 3: (dx,dy,dz)'
                                     '\n- 6: (dx,dy,dz,droll,dpitch,dyaw)'
                                     '\n- 7: (dx,dy,dz,qx,qy,qz,w)'
                                     '\ninstead it is: ', len(action))

            # --- Constraint end-effector pose inside the workspace --- #

            dx, dy, dz = action[:3]
            new_pos = [dx, dy, dz]

            # if orientation is not under control, keep it fixed
            if self._control_orientation == 0:
                #print('hand pose',self._home_hand_pose)
                new_quat_orn = p.getQuaternionFromEuler(self._home_hand_pose[3:6])

            #otherwise, if it is defined as euler angles
            elif len(action) == 6:
                droll, dpitch, dyaw = action[3:]

                eu_orn = [min(m.pi, max(-m.pi, droll)),
                          min(m.pi, max(-m.pi, dpitch)),
                          min(m.pi, max(-m.pi, dyaw))]

                new_quat_orn = p.getQuaternionFromEuler(eu_orn)

            # otherwise, if it is define as quaternion
            elif len(action) == 7:
                new_quat_orn = action[3:7]

            # otherwise, use current orientation
            else:
                new_quat_orn = p.getLinkState(self._panda, self.end_eff_idx)[5]

            #print('new_pos and quat', new_pos, new_quat_orn)
            # --- compute joint positions with IK --- #
            jointPoses = p.calculateInverseKinematics(self._panda, self.end_eff_idx, new_pos, new_quat_orn,
                                                      maxNumIterations=100,
                                                      residualThreshold=.001)
            #print('JointPoses',jointPoses)

            # --- set joint control --- #
            if max_vel == -1:
                p.setJointMotorControlArray(bodyUniqueId=self._panda,
                                            jointIndices=self._joint_name_to_ids.values(),
                                            controlMode=p.POSITION_CONTROL,
                                            targetPositions=jointPoses,
                                            positionGains=[0.2] * len(jointPoses),
                                            velocityGains=[1.0] * len(jointPoses))

            else:
                for i in range(self._num_dof):
                    p.setJointMotorControl2(bodyUniqueId=self._panda,
                                            jointIndex=i,
                                            controlMode=p.POSITION_CONTROL,
                                            targetPosition=jointPoses[i],
                                            maxVelocity=max_vel)

        else:
            # --------------------- #
            # --- Joint control --- #
            # --------------------- #

            assert len(action) == self.joint_action_space, ('number of motor commands differs from number of motor to control', len(action))

            joint_idxs = tuple(self._joint_name_to_ids.values())
            for i, val in enumerate(action):
                motor = joint_idxs[i]
                new_motor_pos = min(self.ul[i], max(self.ll[i], val))

                p.setJointMotorControl2(self.robot_id,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=new_motor_pos,
                                        positionGain=0.5, velocityGain=1.0)

    def pre_grasp(self):
        self.apply_action_fingers([0.04, 0.04], force=5)

    def grasp(self, obj_id=None):
        self.apply_action_fingers([0.015, 0.015], obj_id, force=20)

    def grasp1(self, obj_id=None):
        self.apply_action_fingers([0, 0], obj_id, force=5)

    def grasp2(self, obj_id=None):
        self.apply_action_fingers([0, 0], obj_id, force=5)

    def apply_action_fingers(self, action_grip, obj_id=None, force=0):
        # move finger joints in position control
        # print('force', force)
        assert len(action_grip) == 2, ('finger joints are 2! The number of actions you passed is ', len(action))
        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        # use object id to check contact force and eventually stop the finger motion
        if obj_id is not None:
            _, forces = self.check_contact_fingertips(obj_id)
            # print("contact forces {}".format(forces))
            if forces[0] >= 20.0:
                action_grip[0] = p.getJointState(self._panda, idx_fingers[0])[0]

            if forces[1] >= 20.0:
                action_grip[1] = p.getJointState(self._panda, idx_fingers[1])[0]

        for i, idx in enumerate(idx_fingers):
            p.setJointMotorControl2(self._panda,
                                    idx,
                                    p.POSITION_CONTROL,
                                    targetPosition=action_grip[i],
                                    force=force,
                                    maxVelocity=1.0)

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to control if they are correctly touching an object

        idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]

        p0 = p.getContactPoints(obj_id, self._panda, linkIndexB=idx_fingers[0])
        p1 = p.getContactPoints(obj_id, self._panda, linkIndexB=idx_fingers[1])

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = p.getLinkState(self._panda, idx_fingers[0])[4:6]
            f0_pos_w = p.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = p.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = p.getLinkState(self._panda, idx_fingers[1])[4:6]
            f1_pos_w = p.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = p.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p1_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)

    def get_action_dim(self):
        if not self._use_IK:
            return self.joint_action_space

        if self._control_orientation == 1 and self._control_eu_or_quat == 0:
            print('act_dim',6)
            return 7  # position x,y,z + roll/pitch/yaw of hand frame

        elif self._control_orientation and self._control_eu_or_quat == 1:
            print('act_dim', 7)
            return 7  # position x,y,z + quat of hand frame

        else:
            print('act_dim',3)
            return 4  # position x,y,z

    def get_params(self):
        labels = ['_action_scale', '_action_repeat',
                  '_timestep', '_solver_iterations',
                  '_gripper_bounds', '_pos_low', '_pos_high', '_id']
        params = {label: getattr(self, label) for label in labels}
        return params

    @property
    def parallel(self):
        return False

    def check_params(self, other):
        params = self.get_params()
        assert set(params.keys()) == set(other.keys())
        for key, val in params.items():
            if val != other[key]:
                message = 'Found mismatch in {} | env : {} | demos : {}'.format(
                    key, val, other[key]
                )
                raise RuntimeError(message)

    # def get_constructor(self):
    #     return lambda: self.__class__(*self.args_, **self.kwargs_)

    def reset(self):

        bullet.reset()
        bullet.setup_headless(self._timestep, solver_iterations=self._solver_iterations)
        self._load_meshes()
        self._format_state_query()
        self._prev_pos = np.array(self._pos_init)

        return self.get_observation()

    # def set_reset_hook(self, fn=lambda env: None):
    #     self._reset_hook = fn

    def open_gripper(self, act_repeat=10):
        delta_pos = [0,0,0]
        gripper = 0
        for _ in range(act_repeat):
            self.step(delta_pos, gripper)

    def get_body(self, name):
        if name == 'panda':
            return self._panda
        else:
            return self._objects[name]

    def get_object_midpoint(self, object_key):
        return bullet.get_midpoint(self._objects[object_key])

    def get_end_effector_pos(self):
        return bullet.get_link_state(self._panda, self._end_effector, 'pos')

    def get_end_effector_theta(self):
        return bullet.get_link_state(self._panda, self._end_effector, 'theta')

    def _load_meshes(self):
        self._panda = bullet.objects.panda_robot()
        self._table = bullet.objects.table()
        self._objects = {}
        self._sensors = {}
        self._workspace = bullet.Sensor(self._panda,
            xyz_min=self._pos_low, xyz_max=self._pos_high,
            visualize=False, rgba=[0,1,0,.1])
        self._end_effector = bullet.get_index_by_attribute(
            self._panda, 'link_name', 'gripper_site')


    def _format_state_query(self):
        ## position and orientation of body root
        bodies = [v for k,v in self._objects.items() if not bullet.has_fixed_root(v)]
        ## position and orientation of link
        links = [(self._panda, self._end_effector)]
        ## position and velocity of prismatic joint
        joints = [(self._panda, None)]
        self._state_query = bullet.format_sim_query(bodies, links, joints)

    def _format_action(self, *action):
        if len(action) == 1:
            delta_pos, gripper = action[0][:-1], action[0][-1]
        elif len(action) == 2:
            delta_pos, gripper = action[0], action[1]
        else:
            raise RuntimeError('Unrecognized action: {}'.format(action))
        return np.array(delta_pos), gripper

    def get_observation(self):
        observation = bullet.get_sim_state(*self._state_query)
        return observation

    def step(self, *action):
        delta_pos, gripper = self._format_action(*action)
        pos = bullet.get_link_state(self._panda, self._end_effector, 'pos')
        pos += delta_pos * self._action_scale
        pos = np.clip(pos, self._pos_low, self._pos_high)

        self._simulate(pos, self.theta, gripper)
        if self._visualize: self.visualize_targets(pos)

        observation = self.get_observation()
        reward = self.get_reward(observation)
        done = self.get_termination(observation)
        self._prev_pos = bullet.get_link_state(self._panda, self._end_effector, 'pos')
        return observation, reward, done, {}

    def _simulate(self, pos, theta, gripper):
        for _ in range(self._action_repeat):
            bullet.panda_position_ik(
                self._panda, self._end_effector,
                pos, theta,
                gripper, gripper_bounds=self._gripper_bounds,
                discrete_gripper=False, max_force=self._max_force
            )
            bullet.step_ik()

    def render(self, mode='rgb_array'):
        img, depth, segmentation = bullet.render(
            self._img_dim, self._img_dim, self._view_matrix, self._projection_matrix)
        return img

    def get_termination(self, observation):
        return False

    def get_reward(self, observation):
        return 0

    def visualize_targets(self, pos):
        bullet.add_debug_line(self._prev_pos, pos)

    def save_state(self, *save_path):
        state_id = bullet.save_state(*save_path)
        return state_id

    def load_state(self, load_path):
        bullet.load_state(load_path)
        obs = self.get_observation()
        return obs

    '''
        prevents always needing a gym adapter in softlearning
        @TODO : remove need for this method
    '''
    def convert_to_active_observation(self, obs):
        return obs


