# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym.torch_utils import *

from isaacgymenvs.utils.torch_jit_utils import *
from ..base.vec_task import VecTask

from scipy.spatial.transform import Rotation as R

DOF_BODY_IDS = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
DOF_OFFSETS = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]


KEY_BODY_NAMES = ["right_hand", "left_hand", "right_foot", "left_foot"]

class ExoskeletonBase(VecTask):

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = config
        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.randomize = self.cfg["task"]["randomize"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.camera_follow = self.cfg["env"].get("cameraFollow", False)
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self._local_root_obs = self.cfg["env"]["localRootObs"]
        self._contact_bodies = self.cfg["env"]["contactBodies"]
        self._termination_height = self.cfg["env"]["terminationHeight"]
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]

        # NUM_OBS = 105#13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        # NUM_ACTIONS = 28#28
        self.cfg["env"]["numObservations"] = 105
        self.cfg["env"]["numActions"] = 30

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt
        
        # get gym GPU state tensors
        _actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        _rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        _contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        _dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)

        sensors_per_env = 2
        self.sensor_tensor = gymtorch.wrap_tensor(_sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        self.dof_force_tensor = gymtorch.wrap_tensor(_dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.actor_root_states = gymtorch.wrap_tensor(_actor_root_state)
        #print(self._root_states.shape) [2*num_envs,13] 2*num_envs bc there are 2 actors in 1 env 
        #env[0] human ,env[0] exo,env[1] human, env
        self.root_states_human=self.actor_root_states[::2].contiguous()
        self.root_states_exo=self.actor_root_states[1::2].contiguous()

        # self.root_state_human=self._root_states[:,:]
        # #!!!!!!!!!!!!! I don't know if it's right, looks suspicious to me
        # self._initial_root_states = torch.cat((self._root_states_human,self._root_states_exo),0)
        self.initial_root_states = self.actor_root_states.clone()

        # create some wrapper tensors for different slices
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor)

        self._dof_pos = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 0] #shape:[num_envs,49]

        self.dof_pos_human = self._dof_pos[:,0:28]
        self.dof_pos_exo = self._dof_pos[:,28:]

        self._dof_vel = self._dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.dof_vel_human = self._dof_vel[:,0:28]
        self.dof_vel_exo = self._dof_vel[:,28:]
 
        self._initial_dof_pos_human = torch.zeros_like(self.dof_pos_human, device=self.device, dtype=torch.float)
        right_shoulder_x_handle = self.gym.find_actor_dof_handle(self.env_handles[0], self.humanoid_handles[0], "right_shoulder_x")
        left_shoulder_x_handle = self.gym.find_actor_dof_handle(self.env_handles[0], self.humanoid_handles[0], "left_shoulder_x")
        self._initial_dof_pos_human[:, right_shoulder_x_handle] = 0.5 * np.pi
        self._initial_dof_pos_human[:, left_shoulder_x_handle] = -0.5 * np.pi

        self._initial_dof_pos_exo = torch.zeros_like(self.dof_pos_exo, device=self.device, dtype=torch.float)
        self._initial_dof_vel_human = torch.zeros_like(self.dof_vel_human, device=self.device, dtype=torch.float)
        self._initial_dof_vel_exo = torch.zeros_like(self.dof_vel_exo, device=self.device, dtype=torch.float)
        
        self._rigid_body_state_tensor = gymtorch.wrap_tensor(_rigid_body_state)
        self._rigid_body_state = self._rigid_body_state_tensor.view(self.num_envs, self.num_bodies, 13)        
        self._rigid_body_pos = self._rigid_body_state[..., 0:3]
        self._rigid_body_rot = self._rigid_body_state[..., 3:7]
        self._rigid_body_vel = self._rigid_body_state[..., 7:10]
        self._rigid_body_ang_vel = self._rigid_body_state[..., 10:13]
        self._contact_forces = gymtorch.wrap_tensor(_contact_force_tensor).view(self.num_envs, self.num_bodies, 3)
        self._contact_forces_human = self._contact_forces[:,0:15]
        self._contact_forces_exo = self._contact_forces[:,15:37]

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        


        # self._pd_action_offset = [0, 30, 0, 0, 20, 0, -135, -120, 0, -160, 135, -120, 0, -160, -45, -80, -25, 160,
        #                           0, 0, 0, 0, 0, 45, -80, 25, 160, 0, 0, 0, 0, 0]
        # self._pd_action_scale = [120, 150, 100, 100, 100, 90, 225, 240, 180, 160, 225, 240, 180, 160, 75, 200, 95, 160,
        #                          60, 60, 80, 0.04*180/np.pi, 30, 75, 200, 95, 160, 60, 60, 80, 0.04*180/np.pi, 30]
        self._pd_action_offset = [0, 30, 0, 0, 20, 0, -135, -120, 0, -160, 135, -120, 0, -160, -45, -80, -25, 160,
                                  0, 0, 0, 45, -80, 25, 160, 0, 0, 0,     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]#
        self._pd_action_scale = [120, 150, 100, 100, 100, 90, 225, 240, 180, 160, 225, 240, 180, 160, 75, 200, 95, 160,
                                 60, 60, 80, 75, 200, 95, 160, 60, 60, 80,     40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40]
        
        for i in range(len(self._pd_action_offset)):
            self._pd_action_offset[i] = self._pd_action_offset[i] / 180 * np.pi * 0.5
            self._pd_action_scale[i] = self._pd_action_scale[i] / 180 * np.pi * 0.5
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        self.external_forces = torch.zeros((self.num_envs, 30, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.external_torques = torch.zeros((self.num_envs, 30, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        #define connecttion between exo and human, will be used in pr_physics_step
        # self.external_forces[:,connect_body_indice,2] = connect_forces
        # connect_pairs=[[['base', gymapi.Vec3(0.12, 0.021, -0.2)],['pelvis', gymapi.Vec3(0.12, 0.021, -0.2)],],
        #                [['hr1', gymapi.Vec3(0.12, 0.021, -0.2)],['right_thigh', gymapi.Vec3(0.12, 0.021, -0.2)]],
        #                [['hl1', gymapi.Vec3(0.12, 0.021, -0.2)],['left_thigh', gymapi.Vec3(0.12, 0.021, -0.2)]],
        #                [['ray', gymapi.Vec3(0.12, 0.021, -0.2)],['right_foot', gymapi.Vec3(0.12, 0.021, -0.2)]],
        #                [['lay', gymapi.Vec3(0.12, 0.021, -0.2)],['left_foot', gymapi.Vec3(0.12, 0.021, -0.2)]]
        #                                       ]

        if self.viewer != None:
            self._init_camera()
            
        return
    
    def print_asset_info(self,asset, name):
        print("======== Asset info %s: ========" % (name))
        num_bodies = self.gym.get_asset_rigid_body_count(asset)
        num_joints = self.gym.get_asset_joint_count(asset)
        num_dofs = self.gym.get_asset_dof_count(asset)
        print("Got %d bodies, %d joints, and %d DOFs" %
            (num_bodies, num_joints, num_dofs))

        # Iterate through bodies
        print("Bodies:")
        for i in range(num_bodies):
            name = self.gym.get_asset_rigid_body_name(asset, i)
            print(" %2d: '%s'" % (i, name))

        # Iterate through joints
        print("Joints:")
        for i in range(num_joints):
            name = self.gym.get_asset_joint_name(asset, i)
            type = self.gym.get_asset_joint_type(asset, i)
            type_name = self.gym.get_joint_type_string(type)
            print(" %2d: '%s' (%s)" % (i, name, type_name))

        # iterate through degrees of freedom (DOFs)
        print("DOFs:")
        for i in range(num_dofs):
            name = self.gym.get_asset_dof_name(asset, i)
            type = self.gym.get_asset_dof_type(asset, i)
            type_name = self.gym.get_dof_type_string(type)
            print(" %2d: '%s' (%s)" % (i, name, type_name))
    
    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params) # type: ignore

        return

    def reset_idx(self, env_ids):
        self._reset_actors(env_ids)
        self._refresh_sim_tensors()
        self._compute_observations(env_ids)
        return

    def set_char_color(self, col):
        for i in range(self.num_envs):
            env_ptr = self.env_handles[i]
            handle = self.humanoid_handles[i]

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL, # type: ignore
                                              gymapi.Vec3(col[0], col[1], col[2])) # type: ignore

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)
        return

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        human_file = "mjcf/amp_humanoid.xml"
        exo_file = "urdf/ziqexo/urdf/ziqexo.urdf"
        # if "asset" in self.cfg["env"]:
        #     #asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
        #     asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.fix_base_link=True
        asset_options.max_angular_velocity = 100.0
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        humanoid_asset = self.gym.load_asset(self.sim, asset_root, human_file, asset_options)

        asset_options.fix_base_link = False
        exo_asset = self.gym.load_asset(self.sim, asset_root, exo_file, asset_options)


        # self.print_asset_info(exo_asset, "exo")
        # self.print_asset_info(humanoid_asset, "human")
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]
        
        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()

        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies_human = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_bodies_exo = self.gym.get_asset_rigid_body_count(exo_asset)
        self.num_bodies = self.num_bodies_human + self.num_bodies_exo 
      
        self.num_dof_human = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_dof_exo = self.gym.get_asset_dof_count(exo_asset)
        self.num_dof = self.num_dof_human+self.num_dof_exo
        #
        #
        self.num_joints_human = self.gym.get_asset_joint_count(humanoid_asset)
        self.num_joints_exo = self.gym.get_asset_joint_count(exo_asset)
        self.num_joints = self.num_joints_human+self.num_joints_exo
        # humanoid_start_pose = gymapi.Transform()
        # humanoid_start_pose.p = gymapi.Vec3(*get_axis_params(0.89, self.up_axis_idx))
        # humanoid_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        humanoid_start_pose = gymapi.Transform()
        # humanoid_start_pose.p = gymapi.Vec3(0, 0, 2)
        humanoid_start_pose.p = gymapi.Vec3(0, 0, 0.89)
        exo_start_pose = gymapi.Transform()
        # exo_start_pose.p = gymapi.Vec3(-0.12, -0.021, 2.2)
        exo_start_pose.p = gymapi.Vec3(-0.1, -0.021, 1.0)
        exo_start_pose.r = gymapi.Quat(0,0,0,1)
        # exo_start_pose.r = gymapi.Quat(0,0.5,0,0.869)
        self.start_rotation = torch.tensor([humanoid_start_pose.r.x, humanoid_start_pose.r.y, humanoid_start_pose.r.z, humanoid_start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.exo_handles = []
        self.human_attractor_handles = []
        self.exo_attractor_handles = []

        self.env_handles = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []

        dof_prop_exo = self.gym.get_asset_dof_properties(exo_asset)
        dof_prop_exo['driveMode'][:] = gymapi.DOF_MODE_NONE

        dof_prop_exo['effort'][:] = 0

        dof_prop_exo['stiffness'][:] = 0.0
        dof_prop_exo['damping'][:] = 0.0
        dof_prop_exo['lower'][:] = -1.0
        dof_prop_exo['upper'][:] = 1.0
        dof_prop_exo['armature'][:] = 0.0
        dof_prop_exo['friction'][:] = 0.0
        # dof_prop_exo['driveMode'][5] = gymapi.DOF_MODE_POS
        # dof_prop_exo['stiffness'][5] = 500.0
        # dof_prop_exo['damping'][5] = 40.0
        # dof_prop_exo['driveMode'][11] = gymapi.DOF_MODE_POS
        # dof_prop_exo['stiffness'][11] = 500.0
        # dof_prop_exo['damping'][11] = 40.0

        
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            contact_filter = 1 #1=off 
                    
            human_handle = self.gym.create_actor(env_handle, humanoid_asset, humanoid_start_pose, "humanoid", i, contact_filter, 0)
            exo_handle = self.gym.create_actor(env_handle, exo_asset, exo_start_pose, 'exo', i, contact_filter, 0)
            
            self.gym.set_actor_scale(env_handle,exo_handle,1)
            self.gym.enable_actor_dof_force_sensors(env_handle, human_handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_handle, human_handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.4706, 0.549, 0.6863))

            self.env_handles.append(env_handle)
            self.humanoid_handles.append(human_handle)
            self.exo_handles.append(exo_handle)

            self.gym.set_actor_dof_properties(env_handle, exo_handle, dof_prop_exo)

            if (self._pd_control):
                dof_prop_human = self.gym.get_asset_dof_properties(humanoid_asset)
                dof_prop_human["driveMode"] = gymapi.DOF_MODE_POS
                self.gym.set_actor_dof_properties(env_handle, human_handle, dof_prop_human)

        dof_prop_human = self.gym.get_actor_dof_properties(env_handle, human_handle)
        for j in range(self.num_dof_human):
            if dof_prop_human['lower'][j] > dof_prop_human['upper'][j]:
                self.dof_limits_lower.append(dof_prop_human['upper'][j])
                self.dof_limits_upper.append(dof_prop_human['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop_human['lower'][j])
                self.dof_limits_upper.append(dof_prop_human['upper'][j])
                                    
        # dof_prop = self.gym.get_actor_dof_properties(env_handle, exo_handle)
        # for j in range(self.num_dof_exo):
        #     if dof_prop['lower'][j] > dof_prop['upper'][j]:
        #         self.dof_limits_lower.append(dof_prop['upper'][j])
        #         self.dof_limits_upper.append(dof_prop['lower'][j])
        #     else:
        #         self.dof_limits_lower.append(dof_prop['lower'][j])
        #         self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self._key_body_ids = self._build_key_body_ids_tensor(env_handle, human_handle)
        self._contact_body_ids = self._build_contact_body_ids_tensor(env_handle, human_handle)
        
        # if (self._pd_control):
        #     self._build_pd_action_offset_scale()

        return


    def _compute_reward(self, actions):
        self.rew_buf[:] = compute_humanoid_reward(self.obs_buf)
        return

    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                   self._contact_forces_human, self._contact_body_ids,
                                                   self._rigid_body_pos, self.max_episode_length,
                                                   self._enable_early_termination, self._termination_height)
        return

    def _refresh_sim_tensors(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        return

    def _compute_observations(self, env_ids=None):
        obs = self._compute_joint_obs(env_ids)

        if (env_ids is None):
            self.obs_buf[:] = obs
        else:
            self.obs_buf[env_ids] = obs

        return

    def _compute_joint_obs(self, env_ids=None):
        if (env_ids is None):
            root_states_human = self.root_states_human
            dof_pos_human = self.dof_pos_human
            dof_vel_human = self.dof_vel_human
            key_body_pos_human = self._rigid_body_pos[:, self._key_body_ids, :]
        else:
            root_states_human = self.root_states_human[env_ids]
            dof_pos_human = self.dof_pos_human[env_ids]
            dof_vel_human = self.dof_vel_human[env_ids]
            key_body_pos_human = self._rigid_body_pos[env_ids][:, self._key_body_ids, :]
        

        obs = compute_joint_observations(root_states_human, dof_pos_human, dof_vel_human,
                                            key_body_pos_human, self._local_root_obs)
        return obs

    def _reset_actors(self, env_ids):#?this section is complicated with amp file
        #self._initial_root_states shape [2*num_envs,13] first dimension human follow exo, second dimension pos rot vel angvel
        reset_actor_indices=torch.cat((2*env_ids,2*env_ids+1))
        self.actor_root_states[env_ids] = self.initial_root_states[env_ids]
        self.dof_pos_human[env_ids] = self._initial_dof_pos_human[env_ids]
        self.dof_vel_human[env_ids] = self._initial_dof_vel_human[env_ids]
        self.dof_pos_exo[env_ids] = self._initial_dof_pos_exo[env_ids]
        self.dof_vel_exo[env_ids] = self._initial_dof_vel_exo[env_ids]

        reset_actor_indices_int32 = (reset_actor_indices.to(dtype=torch.int32))

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))
        
        #99.99% sure that the indices is caculated wrong from the very original
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))

        # self.gym.refresh_actor_root_state_tensor(self.sim)
        # print("initial:",self.initial_root_states[0,:],"\ncurrent",self.actor_root_states[0,:],"\n")     
        
        self.external_forces = torch.zeros((self.num_envs, 30, 3), dtype=torch.float32, device=self.device, requires_grad=False)
        self.external_torques = torch.zeros((self.num_envs, 30, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return

    def pre_physics_step(self, actions):

        human_torso_pos=self._rigid_body_pos[:,1,:]
        human_torso_vel=self._rigid_body_vel[:,1,:]

        human_torso_quat=self._rigid_body_rot[:,1,:]
        human_torso_angvel=self._rigid_body_ang_vel[:,1,:]
        offset_torso=torch.tensor([-0.1, 0.0, 0.0],device=human_torso_pos.device).repeat(human_torso_pos.shape[0],1)
        offseted_torso = human_torso_pos + quat_rotate(human_torso_quat, offset_torso)

        human_rthigh_pos=self._rigid_body_pos[:,9,:]
        human_rthigh_quat=self._rigid_body_rot[:,9,:]
        offset_rthigh=torch.tensor([-0.1, 0.0, 0.0],device=human_torso_pos.device).repeat(human_torso_pos.shape[0],1)
        offseted_rthigh=human_torso_pos + quat_rotate(human_torso_quat, offset_torso)

        human_lthigh_pos=self._rigid_body_pos[:,12,:]
        human_lthigh_quat=self._rigid_body_rot[:,12,:]
        offset_lthigh=torch.tensor([-0.1, 0.0, 0.0],device=human_torso_pos.device).repeat(human_torso_pos.shape[0],1)
        offseted_lthigh=human_torso_pos + quat_rotate(human_torso_quat, offset_torso)

        exo_base_pos=self._rigid_body_pos[:,15,:]
        exo_base_vel=self._rigid_body_vel[:,15,:]
        exo_base_quat=self._rigid_body_rot[:,15,:]
        exo_base_angvel=self._rigid_body_ang_vel[:,15,:]

        exo_rlink_pos=self._rigid_body_pos[:,17,:]
        exo_rlink_quat=self._rigid_body_rot[:,17,:]
        exo_llink_pos=self._rigid_body_pos[:,16,:]
        exo_llink_quat=self._rigid_body_rot[:,16,:]

        vec_pos_torso2base=offseted_torso-exo_base_pos
        vec_vel_torso2base=human_torso_vel - exo_base_vel
        dist_torso2base = torch.sqrt(torch.sum(vec_pos_torso2base ** 2, dim=1)) 
        self.external_forces[:, 1, :] = -vec_pos_torso2base * 500 - 60*vec_vel_torso2base
        self.external_forces[:, 15, :] = vec_pos_torso2base * 500 + 60*vec_vel_torso2base

        base_rot_error = quat_mul(human_torso_quat, quat_conjugate(exo_base_quat))
        base_rotvel_error = human_torso_angvel - exo_base_angvel
        # print(human_torso_angvel.shape, exo_base_angvel.shape)
        # input()
        base_rotvel_error = quat_mul(human_torso_angvel, quat_conjugate(exo_base_angvel))
        angle, axis_rot = quat_to_angle_axis(base_rot_error)
        angle_expanded=angle.unsqueeze(-1)
        # angle_vel, axis_rot_vel = quat_to_angle_axis(base_rotvel_error)

        # print(axis_rot.shape, angle.shape, base_rotvel_error.shape)
        # input()
        base_torque = 20 * axis_rot * angle_expanded + 0.4 * base_rotvel_error

        self.external_torques[:, 1, :] = - base_torque
        self.external_torques[:, 15, :] = + base_torque

        quat_torso2base = R.from_quat(human_torso_quat) * R.from_quat(exo_base_quat).inv()
        angle_axis = quat_torso2base.as_rotvec()
        print(angle_axis.shape)
        self.external_forces = torch.zeros((self.num_envs, 10 * 3, 3), dtype=torch.float32, device=self.device, requires_grad=False)

        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.external_forces), gymtorch.unwrap_tensor(self.external_torques), gymapi.ENV_SPACE)
        '''k'''
        #apply action to each DOF
        self.actions = actions.to(self.device).clone()  #shape:[num_envs,NUM_ACTIONS]
        self.actions_exo=torch.zeros(self.actions.shape[0],12,device=self.device)
        
        self.actions_exo[:,5]=self.actions[:,28]
        self.actions_exo[:,11]=self.actions[:,29]
        self.reshaped_actions=torch.cat((self.actions[:,0:28], self.actions_exo), dim = 1)
        pd_tar = self._pd_action_offset + self._pd_action_scale * self.reshaped_actions
        
        pd_tar_tensor = gymtorch.unwrap_tensor(pd_tar)  #unreadable after unwrap 
        # self.gym.set_dof_position_target_tensor(self.sim, pd_tar_tensor)  

        # #draw_lines for debugging
        # self.gym.clear_lines(self.viewer)
        # sphere_rot = gymapi.Quat.from_euler_zyx(0.5 * 3.14, 0, 0)
        # sphere_pose = gymapi.Transform(r=sphere_rot)
        
        # torso_pos_vec = gymapi.Vec3()

        # torso_pos_vec.x, torso_pos_vec.y, torso_pos_vec.z = offseted_torso[0][0].item(), offseted_torso[0][1].item(), offseted_torso[0][2].item()

        # test_point=gymapi.Transform(torso_pos_vec,gymapi.Quat(0.0,0.0,0.0,0.0))
        # # test_point=gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0),gymapi.Quat(0.0,0.0,0.0,0.0))
        # sphere_geom = gymutil.WireframeSphereGeometry(0.09, 12, 12, sphere_pose, color=(0, 0, 1))
        # gymutil.draw_lines(sphere_geom,self.gym,self.viewer,self.env_handles[0],test_point)

        return

    def post_physics_step(self):
        #print(self.obs_buf[:,9][0])
        self.progress_buf += 1

        self._refresh_sim_tensors()
        self._compute_observations()
        self._compute_reward(self.actions)
        self._compute_reset()
        
        self.extras["terminate"] = self._terminate_buf

        # debug viz
        if self.viewer and self.debug_viz:
            self._update_debug_viz()

        return

    def render(self):
        if self.viewer and self.camera_follow:
            self._update_camera()

        super().render()
        return

    def _build_key_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in KEY_BODY_NAMES:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, env_ptr, actor_handle):
        body_ids = []
        for body_name in self._contact_bodies:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar

    def _init_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self._cam_prev_char_pos = self.root_states_human[0, 0:3].cpu().numpy()
        
        cam_pos = gymapi.Vec3(self._cam_prev_char_pos[0] + 2.0, 
                              self._cam_prev_char_pos[1] - 7.0, 
                              1.0)
        cam_target = gymapi.Vec3(self._cam_prev_char_pos[0],
                                 self._cam_prev_char_pos[1],
                                 1.0)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        return

    def _update_camera(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        char_root_pos = self.root_states_human[0, 0:3].cpu().numpy()
        
        cam_trans = self.gym.get_viewer_camera_transform(self.viewer, None)
        cam_pos = np.array([cam_trans.p.x, cam_trans.p.y, cam_trans.p.z])
        cam_delta = cam_pos - self._cam_prev_char_pos

        new_cam_target = gymapi.Vec3(char_root_pos[0], char_root_pos[1], 1.0)
        new_cam_pos = gymapi.Vec3(char_root_pos[0] + cam_delta[0], 
                                  char_root_pos[1] + cam_delta[1], 
                                  cam_pos[2])

        self.gym.viewer_camera_look_at(self.viewer, None, new_cam_pos, new_cam_target)

        self._cam_prev_char_pos[:] = char_root_pos
        return

    def _update_debug_viz(self):
        self.gym.clear_lines(self.viewer)
        return

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def dof_to_obs(pose):
    # type: (Tensor) -> Tensor
    #dof_obs_size = 64
    #dof_offsets = [0, 3, 6, 9, 12, 13, 16, 19, 20, 23, 24, 27, 30, 31, 34]
    dof_obs_size = 52
    dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if (dof_size == 3):
            joint_pose_q = exp_map_to_quat(joint_pose)
            joint_dof_obs = quat_to_tan_norm(joint_pose_q)
            dof_obs_size = 6
        else:
            joint_dof_obs = joint_pose
            dof_obs_size = 1

        dof_obs[:, dof_obs_offset:(dof_obs_offset + dof_obs_size)] = joint_dof_obs
        dof_obs_offset += dof_obs_size

    return dof_obs

@torch.jit.script
def compute_joint_observations(root_states_human, dof_pos_human, dof_vel_human, key_body_pos_human, local_root_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool) -> Tensor
    root_pos = root_states_human[:, 0:3]
    root_rot = root_states_human[:, 3:7]
    root_vel = root_states_human[:, 7:10]
    root_ang_vel = root_states_human[:, 10:13]

    root_h = root_pos[:, 2:3]
    heading_rot = calc_heading_quat_inv(root_rot)

    if (local_root_obs):
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = quat_to_tan_norm(root_rot_obs)

    local_root_vel = my_quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = my_quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos_human - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = my_quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])

    dof_obs = dof_to_obs(dof_pos_human)

    obs = torch.cat((root_h, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel_human, flat_local_key_pos), dim=-1)
    return obs

@torch.jit.script
def compute_humanoid_reward(obs_buf):
    # type: (Tensor) -> Tensor
    reward = torch.ones_like(obs_buf[:, 0]) #-obs_buf[:,9] 
    return reward

@torch.jit.script

#2024-1-24: the reset function seems triggered in a quite... amusing way
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_height):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, float, bool, float) -> Tuple[Tensor, Tensor]
    terminated = torch.zeros_like(reset_buf)

    if (enable_early_termination):
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(masked_contact_buf > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)
        #check this twice
        body_height = rigid_body_pos[..., 2]
        fall_height = body_height < termination_height
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)
        has_fallen = torch.logical_and(fall_contact, fall_height)
        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated