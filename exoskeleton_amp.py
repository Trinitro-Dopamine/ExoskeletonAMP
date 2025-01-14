from enum import Enum
import numpy as np
import torch
import os

from gym import spaces

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgymenvs.tasks.amp.exoskeleton_base import ExoskeletonBase, dof_to_obs
from isaacgymenvs.tasks.amp.utils_amp import gym_util
from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib

from isaacgym.torch_utils import *
from isaacgymenvs.utils.torch_jit_utils import *


#NUM_AMP_OBS_PER_STEP = 105 #13 + 52 + 28 + 12 # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
NUM_AMP_OBS_PER_STEP = 105

class ExoskeletonAMP(ExoskeletonBase):

    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = ExoskeletonAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        motion_file = cfg['env'].get('motion_file', "amp_humanoid_run.npy") #(file_name,default?)
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * NUM_AMP_OBS_PER_STEP

        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None

        return

    def post_physics_step(self):
        super().post_physics_step()
        
        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    def fetch_amp_obs_demo(self, num_samples): # type: ignore
        return self.task.fetch_amp_obs_demo(num_samples)

    def fetch_amp_obs_demo(self, num_samples):
        dt = self.dt
        motion_ids = self._motion_lib.sample_motions(num_samples) 
        if (self._amp_obs_demo_buf is None):
            self._build_amp_obs_demo_buf(num_samples)
        else:
            assert(self._amp_obs_demo_buf.shape[0] == num_samples)
            
        motion_times0 = self._motion_lib.sample_time(motion_ids)
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps])
        motion_times = np.expand_dims(motion_times0, axis=-1)
        time_steps = -dt * np.arange(0, self._num_amp_obs_steps)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        #root_states:[8192, 13]  dof_pos: [8192, 28]  dof_vel: [8192, 28]  key_pos: [8192, 4, 3]
        # print(self._amp_obs_demo_buf.shape)
        self._amp_obs_demo_buf[:] = amp_obs_demo.view(self._amp_obs_demo_buf.shape) #_amp_obs_demo_buf=[4096,2,126]

        amp_obs_demo_flat = self._amp_obs_demo_buf.view(-1, self.get_num_amp_obs())
        return amp_obs_demo_flat

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, NUM_AMP_OBS_PER_STEP), device=self.device, dtype=torch.float)
        return
        

    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file, 
                                     num_dofs=self.num_dof_human,
                                     key_body_ids=self._key_body_ids.cpu().numpy(), 
                                     device=self.device)
        return
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)
        return

    def _reset_actors(self, env_ids):
        if (self._state_init == ExoskeletonAMP.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == ExoskeletonAMP.StateInit.Start
            or self._state_init == ExoskeletonAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == ExoskeletonAMP.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0

        return
    
    def _reset_default(self, env_ids):
        # self.dof_pos_human[env_ids] = self._initial_dof_pos_human[env_ids]
        # self.dof_vel_human[env_ids] = self._initial_dof_vel_human[env_ids]
        
        # self.dof_pos_exo[env_ids] = self._initial_dof_pos_exo[env_ids]
        # self.dof_vel_exo[env_ids] = self._initial_dof_vel_exo[env_ids]

        # env_ids_int32 = env_ids.to(dtype=torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        # self._reset_default_env_ids = env_ids
        # return
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
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self._terminate_buf[env_ids] = 0
        return
    
    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        motion_ids = self._motion_lib.sample_motions(num_envs)

        if (self._state_init == ExoskeletonAMP.StateInit.Random
            or self._state_init == ExoskeletonAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif (self._state_init == ExoskeletonAMP.StateInit.Start):
            motion_times = np.zeros(num_envs)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        

        self._set_env_state(env_ids=env_ids, 
                            root_pos=root_pos, 
                            root_rot=root_rot, 
                            dof_pos=dof_pos, 
                            root_vel=root_vel, 
                            root_ang_vel=root_ang_vel, 
                            dof_vel=dof_vel)

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if (len(ref_reset_ids) > 0):
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if (len(default_reset_ids) > 0):
            self._reset_default(default_reset_ids)
        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if (len(self._reset_default_env_ids) > 0):
            self._init_amp_obs_default(self._reset_default_env_ids)

        if (len(self._reset_ref_env_ids) > 0):
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        # print("\n\n\n",curr_amp_obs,self._curr_amp_obs_buf)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = np.tile(np.expand_dims(motion_ids, axis=-1), [1, self._num_amp_obs_steps - 1])
        motion_times = np.expand_dims(motion_times, axis=-1)
        time_steps = -dt * (np.arange(0, self._num_amp_obs_steps - 1) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.flatten()
        motion_times = motion_times.flatten()
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        root_states = torch.cat([root_pos, root_rot, root_vel, root_ang_vel], dim=-1)
        amp_obs_demo = build_amp_observations(root_states, dof_pos, dof_vel, key_pos,
                                      self._local_root_obs)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return
    
    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        # self.actor_root_states[env_ids] = self.initial_root_states[env_ids]
        # self.dof_pos_human[env_ids] = self._initial_dof_pos_human[env_ids]
        # self.dof_vel_human[env_ids] = self._initial_dof_vel_human[env_ids]
        # self.dof_pos_exo[env_ids] = self._initial_dof_pos_exo[env_ids]
        # self.dof_vel_exo[env_ids] = self._initial_dof_vel_exo[env_ids]

        # reset_actor_indices_int32 = (reset_actor_indices.to(dtype=torch.int32))

        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.initial_root_states),
        #                                              gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))     
        # self.gym.set_dof_state_tensor_indexed(self.sim,
        #                                       gymtorch.unwrap_tensor(self._dof_state),
        #                                       gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))

        reset_actor_indices=torch.cat((2*env_ids, 2*env_ids+1))
        #human 
        self.actor_root_states[2*env_ids, 0:3] = root_pos.clone()
        self.actor_root_states[2*env_ids, 3:7] = root_rot.clone()
        self.actor_root_states[2*env_ids, 7:10] = root_vel.clone()
        self.actor_root_states[2*env_ids, 10:13] = root_ang_vel.clone()
        #exo 
        self.actor_root_states[2*env_ids+1, 0:3] = root_pos.clone()
        self.actor_root_states[2*env_ids+1, 3:7] = root_rot.clone()
        self.actor_root_states[2*env_ids+1, 7:10] = root_vel.clone()
        self.actor_root_states[2*env_ids+1, 10:13] = root_ang_vel.clone()
        
        #print(self.actor_root_states.shape) shape:[2*num_envs,13]
        #self._dof_state [196,2]
        #error:dof_state is not reset to desired random time stamp
        #self._dof_pos[env_ids][:,0:28] = dof_pos

        self._dof_pos[env_ids,0:28] = dof_pos.clone()
        self._dof_pos[env_ids,28:] = 0
        self._dof_vel[env_ids,0:28] = dof_vel.clone()
        self._dof_vel[env_ids,28:] = 0

        env_ids_int32 = env_ids.to(dtype=torch.int32)

        reset_actor_indices_int32 = (reset_actor_indices.to(dtype=torch.int32))
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.actor_root_states), 
                                                    gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                                    gymtorch.unwrap_tensor(reset_actor_indices_int32), len(reset_actor_indices_int32))
        return

    def _update_hist_amp_obs(self, env_ids=None):
        if (env_ids is None):
            self._hist_amp_obs_buf[:] = self._amp_obs_buf[:, 0:(self._num_amp_obs_steps - 1)]
        else:
            self._hist_amp_obs_buf[env_ids] = self._amp_obs_buf[env_ids, 0:(self._num_amp_obs_steps - 1)]
        return
    
    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        # print(self._root_states.shape)
        # print("\n\n_root_states:",self._root_states_human.shape,"_dof_pos:",self._dof_pos.shape,"_dof_vel:",self._dof_vel.shape,
        #       "key_body_pos:",key_body_pos.shape,"_local_root_obs:",self._local_root_obs)
        #_root_states:[512, 13]  _dof_pos:[256, 49]  _dof_vel:[256, 49]  key_body_pos:[256, 4, 3]  _local_root_obs: False
        #sum:
        if (env_ids is None):
            self._curr_amp_obs_buf[:] = build_amp_observations(self.root_states_human, self.dof_pos_human, self.dof_vel_human, key_body_pos,
                                                                self._local_root_obs)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self.root_states_human[env_ids], self.dof_pos_human[env_ids], 
                                                                    self.dof_vel_human[env_ids], key_body_pos[env_ids],
                                                                    self._local_root_obs)
        return


#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def build_amp_observations(root_states_human, dof_pos_human, dof_vel_human, key_body_pos_human, local_root_obs):
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
    # print("root_h:",root_h.shape,"root_rot_obs:",root_rot_obs.shape,"local_root_vel:",local_root_vel.shape,"local_root_ang_vel:",local_root_ang_vel.shape
    #       ,"dof_obs:",dof_obs.shape,"dof_vel:",dof_vel.shape,"flat_local_key_pos:",flat_local_key_pos.shape)
    return obs