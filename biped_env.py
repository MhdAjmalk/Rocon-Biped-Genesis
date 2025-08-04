import torch
import math
import genesis as gs
from genesis.utils.geom import quat_to_xyz, transform_by_quat, inv_quat, transform_quat_by_quat
from genesis.sensors import RigidContactForceGridSensor
import numpy as np


def gs_rand_float(lower, upper, shape, device):
    return (upper - lower) * torch.rand(size=shape, device=device) + lower


class BipedEnv:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, show_viewer=False):
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_privileged_obs = None
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device

        self.simulate_action_latency = True  # there is a 1 step latency on real robot
        self.dt = 0.02  # control frequency on real robot is 50hz
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg

        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        # create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
            show_viewer=show_viewer,
        )

        # add plain
        self.scene.add_entity(gs.morphs.URDF(file="urdf/plane/plane.urdf", fixed=True))

        # add robot
        self.base_init_pos = torch.tensor(self.env_cfg["base_init_pos"], device=gs.device)
        self.base_init_quat = torch.tensor(self.env_cfg["base_init_quat"], device=gs.device)
        self.inv_base_init_quat = inv_quat(self.base_init_quat)
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/biped_v4.urdf",
                pos=self.base_init_pos.cpu().numpy(),
                quat=self.base_init_quat.cpu().numpy(),
            ),
        )

        # build
        self.scene.build(n_envs=num_envs)

        # names to indices
        self.motors_dof_idx = [self.robot.get_joint(name).dof_start for name in self.env_cfg["joint_names"]]
        
        # foot contact sensors
        self.left_foot_contact_sensor = None
        self.right_foot_contact_sensor = None
        
        # Find foot links and create contact sensors
        for link in self.robot.links:
            if link.name == "revolute_leftfoot":
                self.left_foot_contact_sensor = RigidContactForceGridSensor(
                    entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )
            elif link.name == "revolute_rightfoot":
                self.right_foot_contact_sensor = RigidContactForceGridSensor(
                    entity=self.robot, link_idx=link.idx, grid_size=(2, 2, 2)
                )

        # PD control parameters
        self.robot.set_dofs_kp([self.env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([self.env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)

        # Domain randomization setup
        self.orig_kp = torch.tensor([self.env_cfg["kp"]] * self.num_actions, device=gs.device)
        self.randomized_kp = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        if self.env_cfg["domain_rand"]["push_robot"]:
            self.push_interval = math.ceil(self.env_cfg["domain_rand"]["push_interval_s"] / self.dt)

        # prepare reward functions and multiply reward scales by dt
        self.reward_functions, self.episode_sums = dict(), dict()
        for name in self.reward_scales.keys():
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

        # initialize buffers
        self.base_lin_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_ang_vel = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.projected_gravity = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], device=gs.device, dtype=gs.tc_float).repeat(
            self.num_envs, 1
        )
        self.obs_buf = torch.zeros((self.num_envs, self.num_obs), device=gs.device, dtype=gs.tc_float)
        self.rew_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)
        self.reset_buf = torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.episode_length_buf = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
        self.commands = torch.zeros((self.num_envs, self.num_commands), device=gs.device, dtype=gs.tc_float)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        self.actions = torch.zeros((self.num_envs, self.num_actions), device=gs.device, dtype=gs.tc_float)
        self.last_actions = torch.zeros_like(self.actions)
        self.dof_pos = torch.zeros_like(self.actions)
        self.dof_vel = torch.zeros_like(self.actions)
        self.last_dof_vel = torch.zeros_like(self.actions)
        self.base_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)
        self.base_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)
        self.default_dof_pos = torch.tensor(
            [self.env_cfg["default_joint_angles"][name] for name in self.env_cfg["joint_names"]],
            device=gs.device,
            dtype=gs.tc_float,
        )
        
        # Additional buffers for new observations
        self.foot_contacts = torch.zeros((self.num_envs, 2), device=gs.device, dtype=gs.tc_float)  # L/R foot contact
        
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = 0.0  # Set angular velocity to zero (no turning)

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.simulate_action_latency else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        self.robot.control_dofs_position(target_dof_pos, self.motors_dof_idx)
        self.scene.step()

        # update buffers
        self.episode_length_buf += 1
        self.base_pos[:] = self.robot.get_pos()
        self.base_quat[:] = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(torch.ones_like(self.base_quat) * self.inv_base_init_quat, self.base_quat),
            rpy=True,
            degrees=True,
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel[:] = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel[:] = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos[:] = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel[:] = self.robot.get_dofs_velocity(self.motors_dof_idx)
        
        # Update foot contact data
        if self.left_foot_contact_sensor is not None:
            left_contact_data = self.left_foot_contact_sensor.read()
            # Convert to a tensor, move to the correct device, then reshape
            left_contact_tensor = torch.as_tensor(left_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            # Now perform torch operations
            self.foot_contacts[:, 0] = torch.max(torch.norm(left_contact_tensor, dim=-1), dim=-1)[0]

        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            right_contact_tensor = torch.as_tensor(right_contact_data, device=self.device).reshape(self.num_envs, -1, 3)
            self.foot_contacts[:, 1] = torch.max(torch.norm(right_contact_tensor, dim=-1), dim=-1)[0]
        
        # Domain randomization: Apply external perturbations (robot pushing)
        dr_cfg = self.env_cfg["domain_rand"]
        if dr_cfg["push_robot"]:
            # Find which environments should be pushed in this timestep
            push_now_idx = (self.episode_length_buf % self.push_interval == 0).nonzero(as_tuple=False).reshape((-1,))
            
            if len(push_now_idx) > 0:
                # Sample random force direction and magnitude
                max_vel = dr_cfg["max_push_vel_xy"]
                force_vec = gs_rand_float(-max_vel, max_vel, (len(push_now_idx), 2), gs.device)
                
                # Apply impulse to each environment individually
                try:
                    for i, env_idx in enumerate(push_now_idx):
                        # Create 3D force vector with zero z-component
                        force_3d = torch.zeros(3, device=gs.device)
                        force_3d[:2] = force_vec[i]
                        
                        # Apply force to the robot base - Genesis API may vary
                        self.robot.apply_force(force_3d, is_global=True)
                except (AttributeError, TypeError) as e:
                    # If force application methods don't exist, skip and warn once
                    if not hasattr(self, '_force_warning_shown'):
                        print(f"Warning: External force application not supported: {e}")
                        self._force_warning_shown = True
        
        # resample commands
        envs_idx = (
            (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0)
            .nonzero(as_tuple=False)
            .reshape((-1,))
        )
        self._resample_commands(envs_idx)

        # check termination and reset
        self.reset_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= torch.abs(self.base_euler[:, 1]) > self.env_cfg["termination_if_pitch_greater_than"]
        self.reset_buf |= torch.abs(self.base_euler[:, 0]) > self.env_cfg["termination_if_roll_greater_than"]

        time_out_idx = (self.episode_length_buf > self.max_episode_length).nonzero(as_tuple=False).reshape((-1,))
        self.extras["time_outs"] = torch.zeros_like(self.reset_buf, device=gs.device, dtype=gs.tc_float)
        self.extras["time_outs"][time_out_idx] = 1.0

        self.reset_idx(self.reset_buf.nonzero(as_tuple=False).reshape((-1,)))

        # compute reward
        self.rew_buf[:] = 0.0
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew

        # compute observations
        # Extract specific joint angles and velocities based on joint order: 
        # [left_hip1, left_hip2, left_knee, left_ankle, right_hip1, right_hip2, right_knee, right_ankle, torso]
        
        # Hip joints (left_hip1, left_hip2, right_hip1, right_hip2)
        hip_angles = torch.cat([
            (self.dof_pos[:, [0, 1]] - self.default_dof_pos[[0, 1]]) * self.obs_scales["dof_pos"],  # Left hip
            (self.dof_pos[:, [4, 5]] - self.default_dof_pos[[4, 5]]) * self.obs_scales["dof_pos"],  # Right hip
        ], dim=1)  # 4 values
        
        hip_velocities = torch.cat([
            self.dof_vel[:, [0, 1]] * self.obs_scales["dof_vel"],  # Left hip
            self.dof_vel[:, [4, 5]] * self.obs_scales["dof_vel"],  # Right hip
        ], dim=1)  # 4 values
        
        # Knee joints (left_knee, right_knee)
        knee_angles = torch.cat([
            (self.dof_pos[:, [2]] - self.default_dof_pos[[2]]) * self.obs_scales["dof_pos"],  # Left knee
            (self.dof_pos[:, [6]] - self.default_dof_pos[[6]]) * self.obs_scales["dof_pos"],  # Right knee
        ], dim=1)  # 2 values
        
        knee_velocities = torch.cat([
            self.dof_vel[:, [2]] * self.obs_scales["dof_vel"],  # Left knee
            self.dof_vel[:, [6]] * self.obs_scales["dof_vel"],  # Right knee
        ], dim=1)  # 2 values
        
        # Ankle joints (left_ankle, right_ankle)
        ankle_angles = torch.cat([
            (self.dof_pos[:, [3]] - self.default_dof_pos[[3]]) * self.obs_scales["dof_pos"],  # Left ankle
            (self.dof_pos[:, [7]] - self.default_dof_pos[[7]]) * self.obs_scales["dof_pos"],  # Right ankle
        ], dim=1)  # 2 values
        
        ankle_velocities = torch.cat([
            self.dof_vel[:, [3]] * self.obs_scales["dof_vel"],  # Left ankle
            self.dof_vel[:, [7]] * self.obs_scales["dof_vel"],  # Right ankle
        ], dim=1)  # 2 values
        
        # Normalize foot contacts (binary or normalized force)
        foot_contacts_normalized = torch.clamp(self.foot_contacts, 0, 1)  # 2 values
        
        self.obs_buf = torch.cat(
            [
                self.base_euler[:, :2] * self.obs_scales.get("base_euler", 1.0),  # Torso pitch/roll angle (2)
                self.base_ang_vel[:, :2] * self.obs_scales["ang_vel"],  # Torso pitch/roll velocity (2)
                self.base_ang_vel[:, [2]] * self.obs_scales["ang_vel"],  # Torso yaw velocity (1)
                self.base_lin_vel[:, :2] * self.obs_scales["lin_vel"],  # Torso linear velocity X,Y (2)
                self.base_pos[:, [2]] * self.obs_scales.get("base_height", 1.0),  # Torso height (1)
                self.commands[:, :3] * self.commands_scale,  # Velocity commands [lin_vel_x, lin_vel_y, ang_vel] (3)
                hip_angles,  # Hip joint angles L/R (4)
                hip_velocities,  # Hip joint velocities L/R (4)
                knee_angles,  # Knee joint angles L/R (2)
                knee_velocities,  # Knee joint velocities L/R (2)
                ankle_angles,  # Ankle joint angles L/R (2)
                ankle_velocities,  # Ankle joint velocities L/R (2)
                foot_contacts_normalized,  # Foot contact L/R (2)
                self.last_actions,  # Previous actions (9)
            ],
            axis=-1,
        )  # Total: 2+2+1+2+1+3+4+4+2+2+2+2+2+9 = 38 observations

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]

        self.extras["observations"]["critic"] = self.obs_buf

        return self.obs_buf, self.rew_buf, self.reset_buf, self.extras

    def get_observations(self):
        self.extras["observations"]["critic"] = self.obs_buf
        return self.obs_buf, self.extras

    def get_privileged_observations(self):
        return None

    def reset_idx(self, envs_idx):
        if len(envs_idx) == 0:
            return

        # --- Domain Randomization ---
        dr_cfg = self.env_cfg["domain_rand"]

        # Randomize motor strength (kp)
        if dr_cfg["randomize_motor_strength"]:
            strength_scale = gs_rand_float(
                dr_cfg["motor_strength_range"][0],
                dr_cfg["motor_strength_range"][1],
                (len(envs_idx),),
                gs.device
            ).unsqueeze(1)  # Shape: (num_envs, 1) for broadcasting
            self.randomized_kp[envs_idx] = self.orig_kp * strength_scale
            
            # Apply kp values for each environment individually
            for i, env_idx in enumerate(envs_idx):
                self.robot.set_dofs_kp(
                    self.randomized_kp[env_idx],  # 1D tensor for single environment
                    self.motors_dof_idx, 
                    envs_idx=[env_idx]
                )

        # Randomize friction
        if dr_cfg["randomize_friction"]:
            try:
                friction = gs_rand_float(
                    dr_cfg["friction_range"][0],
                    dr_cfg["friction_range"][1],
                    (len(envs_idx),),
                    gs.device
                )
                # Set friction for the robot - note that Genesis may not support per-environment friction
                # This is a placeholder that may need adjustment based on actual Genesis API
                for i, env_idx in enumerate(envs_idx):
                    self.robot.set_friction(friction[i].item())
            except (AttributeError, TypeError) as e:
                # If the method doesn't exist or has different signature, skip friction randomization
                if not hasattr(self, '_friction_warning_shown'):
                    print(f"Warning: Friction randomization not supported: {e}")
                    self._friction_warning_shown = True

        # Randomize mass of the torso
        if dr_cfg["randomize_mass"]:
            try:
                # Find torso link - try different possible names
                torso_link = None
                for link in self.robot.links:
                    if link.name in ["torso", "base_link", "torso_link"]:
                        torso_link = link
                        break
                
                if torso_link is not None:
                    base_mass = torso_link.mass  # Get base mass from link properties
                    added_mass = gs_rand_float(
                        dr_cfg["added_mass_range"][0],
                        dr_cfg["added_mass_range"][1],
                        (len(envs_idx),),
                        gs.device
                    )
                    # Apply mass changes - Genesis may not support per-environment mass changes
                    for i, env_idx in enumerate(envs_idx):
                        new_mass = base_mass + added_mass[i].item()
                        self.robot.set_link_mass(torso_link.idx, new_mass)
                else:
                    # If torso link not found, print warning only once
                    if not hasattr(self, '_mass_warning_shown'):
                        print("Warning: Torso link not found for mass randomization")
                        self._mass_warning_shown = True
            except (AttributeError, TypeError) as e:
                # If mass randomization methods don't exist, skip and warn once
                if not hasattr(self, '_mass_api_warning_shown'):
                    print(f"Warning: Mass randomization not supported: {e}")
                    self._mass_api_warning_shown = True

        # --- End of Domain Randomization ---

        # reset dofs
        self.dof_pos[envs_idx] = self.default_dof_pos
        self.dof_vel[envs_idx] = 0.0
        self.robot.set_dofs_position(
            position=self.dof_pos[envs_idx],
            dofs_idx_local=self.motors_dof_idx,
            zero_velocity=True,
            envs_idx=envs_idx,
        )

        # reset base
        self.base_pos[envs_idx] = self.base_init_pos
        self.base_quat[envs_idx] = self.base_init_quat.reshape(1, -1)
        self.robot.set_pos(self.base_pos[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.robot.set_quat(self.base_quat[envs_idx], zero_velocity=False, envs_idx=envs_idx)
        self.base_lin_vel[envs_idx] = 0
        self.base_ang_vel[envs_idx] = 0
        self.robot.zero_all_dofs_velocity(envs_idx)

        # reset buffers
        self.last_actions[envs_idx] = 0.0
        self.last_dof_vel[envs_idx] = 0.0
        self.foot_contacts[envs_idx] = 0.0
        self.randomized_kp[envs_idx] = self.orig_kp  # Reset to original kp values
        self.episode_length_buf[envs_idx] = 0
        self.reset_buf[envs_idx] = True

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = (
                torch.mean(self.episode_sums[key][envs_idx]).item() / self.env_cfg["episode_length_s"]
            )
            self.episode_sums[key][envs_idx] = 0.0

        self._resample_commands(envs_idx)

    def reset(self):
        self.reset_buf[:] = True
        self.reset_idx(torch.arange(self.num_envs, device=gs.device))
        return self.obs_buf, None

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        # Penalize joint poses far away from default pose
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_foot_clearance(self):
        # Penalize feet dragging (for biped locomotion)
        # This is a simplified version - you might want to add foot position tracking
        return torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_forward_velocity(self):
        # Exponential reward for forward velocity tracking
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)
        vel_error = torch.square(self.base_lin_vel[:, 0] - v_target)
        sigma = self.reward_cfg.get("tracking_sigma", 0.25)
        return torch.exp(-vel_error / sigma)

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (forward velocity)
        lin_vel_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (sideways velocity)
        lin_vel_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_alive_bonus(self):
        # Alive bonus: constant positive value per step
        return torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float)

    def _reward_fall_penalty(self):
        # Fall penalty: large negative value on termination
        # This will be applied when robot falls (high roll/pitch angles)
        fall_condition = (
            (torch.abs(self.base_euler[:, 0]) > self.env_cfg.get("fall_roll_threshold", 30.0)) |  # Roll > 30 degrees
            (torch.abs(self.base_euler[:, 1]) > self.env_cfg.get("fall_pitch_threshold", 30.0))   # Pitch > 30 degrees
        )
        return torch.where(
            fall_condition,
            torch.ones((self.num_envs,), device=gs.device, dtype=gs.tc_float),  # Apply penalty
            torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_float)  # No penalty
        )

    def _reward_torso_stability(self):
        # Torso stability: -w_orient * (φ² + θ²) or w_orient * exp(-k(φ² + θ²))
        # Using exponential form for smoother reward
        orientation_error = torch.sum(torch.square(self.base_euler[:, :2]), dim=1)  # φ² + θ² (roll² + pitch²)
        k_stability = self.reward_cfg.get("stability_factor", 1.0)
        return torch.exp(-k_stability * orientation_error)

    def _reward_height_maintenance(self):
        # Height maintenance: -w_height * (z_target - z_current)²
        # Note: This is similar to base_height but with different formulation
        z_target = self.reward_cfg.get("height_target", 0.35)
        height_error = torch.square(z_target - self.base_pos[:, 2])
        return -height_error  # Return negative error (will be scaled by negative weight in config)

    def _reward_joint_movement(self):
        # Reward for joint movement - encourages locomotion
        # Reward based on absolute joint velocities (encourages movement)
        joint_vel_magnitude = torch.sum(torch.abs(self.dof_vel), dim=1)
        movement_threshold = self.reward_cfg.get("movement_threshold", 0.1)
        movement_scale = self.reward_cfg.get("movement_scale", 1.0)
        
        # Give reward proportional to joint movement, but cap it to avoid excessive movement
        return torch.clamp(joint_vel_magnitude * movement_scale, 0.0, movement_threshold)
    
    def _reward_sinusoidal_gait(self):
        """
        Rewards the robot for following a sinusoidal joint trajectory for leg joints only.
        This encourages a rhythmic, gait-like motion without affecting the torso.
        """
        # Get sine wave parameters from config, with defaults
        amplitude = self.reward_cfg.get("gait_amplitude", 0.5)  # rad
        frequency = self.reward_cfg.get("gait_frequency", 0.5)  # Hz
        
        # Define phase offsets for leg joints only (excluding torso at index 8).
        # [R_H1, R_H2, R_K, R_A, L_H1, L_H2, L_K, L_A]
        # We'll make the main hip joints (H1) move opposite to each other.
        phase_offsets = torch.tensor(
            [0, 0, 0, 0, np.pi, 0, 0, 0], 
            device=self.device, dtype=gs.tc_float
        )

        # Calculate the current time in the episode
        time = self.episode_length_buf * self.dt
        time = time.unsqueeze(1) # Reshape for broadcasting

        # Calculate the target angle for leg joints only (exclude torso joint at index 8)
        leg_joints_default = self.default_dof_pos[:-1]  # All joints except the last one (torso)
        target_leg_pos = leg_joints_default + amplitude * torch.sin(
            2 * np.pi * frequency * time + phase_offsets
        )

        # Calculate the error between the current and target joint positions for leg joints only
        leg_joints_current = self.dof_pos[:, :-1]  # All joints except the last one (torso)
        error = torch.sum(torch.square(leg_joints_current - target_leg_pos), dim=1)

        # Use an exponential function to convert the error to a reward
        # A smaller error results in a higher reward.
        sigma = self.reward_cfg.get("gait_sigma", 0.25)
        return torch.exp(-error / sigma)

    def _reward_torso_sinusoidal(self):
        """
        Rewards the torso for following a sinusoidal motion.
        This encourages rhythmic torso movement independent of leg gait.
        """
        # Get torso sine wave parameters from config, with defaults
        torso_amplitude = self.reward_cfg.get("torso_amplitude", 0.2)  # rad (smaller amplitude for torso)
        torso_frequency = self.reward_cfg.get("torso_frequency", 0.3)  # Hz (different frequency from legs)
        torso_phase = self.reward_cfg.get("torso_phase", 0.0)  # Phase offset for torso

        # Calculate the current time in the episode
        time = self.episode_length_buf * self.dt
        time = time.unsqueeze(1) # Reshape for broadcasting

        # Calculate the target angle for torso joint (index 8)
        torso_default = self.default_dof_pos[8]  # Torso joint default position
        target_torso_pos = torso_default + torso_amplitude * torch.sin(
            2 * np.pi * torso_frequency * time + torso_phase
        )

        # Calculate the error between current and target torso position
        torso_current = self.dof_pos[:, 8]  # Current torso joint position
        error = torch.square(torso_current - target_torso_pos.squeeze())

        # Use an exponential function to convert the error to a reward
        sigma = self.reward_cfg.get("torso_sigma", 0.25)
        return torch.exp(-error / sigma)
