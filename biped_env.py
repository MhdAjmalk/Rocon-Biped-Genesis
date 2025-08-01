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
        self.left_foot_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Left foot position
        self.right_foot_pos = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Right foot position
        self.left_foot_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)  # Left foot orientation
        self.right_foot_quat = torch.zeros((self.num_envs, 4), device=gs.device, dtype=gs.tc_float)  # Right foot orientation
        self.left_foot_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Left foot euler angles
        self.right_foot_euler = torch.zeros((self.num_envs, 3), device=gs.device, dtype=gs.tc_float)  # Right foot euler angles
        
        self.extras = dict()  # extra information for logging
        self.extras["observations"] = dict()

    def _resample_commands(self, envs_idx):
        self.commands[envs_idx, 0] = gs_rand_float(*self.command_cfg["lin_vel_x_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 1] = gs_rand_float(*self.command_cfg["lin_vel_y_range"], (len(envs_idx),), gs.device)
        self.commands[envs_idx, 2] = gs_rand_float(*self.command_cfg["ang_vel_range"], (len(envs_idx),), gs.device)

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
            self.foot_contacts[:, 0] = torch.max(torch.norm(left_contact_data.view(self.num_envs, -1, 3), dim=-1), dim=-1)[0]
        if self.right_foot_contact_sensor is not None:
            right_contact_data = self.right_foot_contact_sensor.read()
            self.foot_contacts[:, 1] = torch.max(torch.norm(right_contact_data.view(self.num_envs, -1, 3), dim=-1), dim=-1)[0]
        
        # Update foot position and orientation data
        # In Genesis, we need to access link positions/orientations through the link objects directly
        # since the robot entity doesn't have get_link_pos/get_link_quat methods
        for link in self.robot.links:
            if link.name == "revolute_leftfoot":
                self.left_foot_pos[:] = link.get_pos()
                self.left_foot_quat[:] = link.get_quat()
                self.left_foot_euler[:] = quat_to_xyz(self.left_foot_quat, rpy=True, degrees=True)
            elif link.name == "revolute_rightfoot":
                self.right_foot_pos[:] = link.get_pos()
                self.right_foot_quat[:] = link.get_quat()
                self.right_foot_euler[:] = quat_to_xyz(self.right_foot_quat, rpy=True, degrees=True)

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
                hip_angles,  # Hip joint angles L/R (4)
                hip_velocities,  # Hip joint velocities L/R (4)
                knee_angles,  # Knee joint angles L/R (2)
                knee_velocities,  # Knee joint velocities L/R (2)
                ankle_angles,  # Ankle joint angles L/R (2)
                ankle_velocities,  # Ankle joint velocities L/R (2)
                foot_contacts_normalized,  # Foot contact L/R (2)
                self.left_foot_pos * self.obs_scales.get("foot_pos", 1.0),  # Left foot position (3)
                self.right_foot_pos * self.obs_scales.get("foot_pos", 1.0),  # Right foot position (3)
                self.left_foot_euler * self.obs_scales.get("foot_euler", 1.0),  # Left foot orientation (3)
                self.right_foot_euler * self.obs_scales.get("foot_euler", 1.0),  # Right foot orientation (3)
                self.last_actions,  # Previous actions (9)
            ],
            axis=-1,
        )  # Total: 2+2+1+2+1+4+4+2+2+2+2+2+3+3+3+3+9 = 47 observations

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
        self.left_foot_pos[envs_idx] = 0.0
        self.right_foot_pos[envs_idx] = 0.0
        self.left_foot_quat[envs_idx] = 0.0
        self.right_foot_quat[envs_idx] = 0.0
        self.left_foot_euler[envs_idx] = 0.0
        self.right_foot_euler[envs_idx] = 0.0
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
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

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
        # Forward velocity reward: w_vel * v_x or w_vel * exp(-(v_x - v_target)^2)
        # Using exponential form for smoother reward
        v_target = self.reward_cfg.get("forward_velocity_target", 0.5)  # Target forward velocity
        velocity_error = torch.square(self.base_lin_vel[:, 0] - v_target)
        return torch.exp(-velocity_error / self.reward_cfg.get("velocity_sigma", 0.25))

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

    def _reward_foot_orientation_contact(self):
        # Reward for keeping feet parallel to ground when in contact
        # Only reward when foot is in contact with ground
        contact_threshold = self.reward_cfg.get("contact_threshold", 0.1)
        
        # Check if feet are in contact
        left_in_contact = self.foot_contacts[:, 0] > contact_threshold
        right_in_contact = self.foot_contacts[:, 1] > contact_threshold
        
        # Calculate foot orientation errors (roll and pitch should be near zero)
        left_orientation_error = torch.sum(torch.square(self.left_foot_euler[:, :2]), dim=1)  # roll² + pitch²
        right_orientation_error = torch.sum(torch.square(self.right_foot_euler[:, :2]), dim=1)  # roll² + pitch²
        
        # Only apply reward when foot is in contact
        left_reward = torch.where(
            left_in_contact,
            torch.exp(-left_orientation_error / self.reward_cfg.get("foot_orientation_sigma", 100.0)),
            torch.zeros_like(left_orientation_error)
        )
        
        right_reward = torch.where(
            right_in_contact,
            torch.exp(-right_orientation_error / self.reward_cfg.get("foot_orientation_sigma", 100.0)),
            torch.zeros_like(right_orientation_error)
        )
        
        return left_reward + right_reward

    def _reward_foot_distance_penalty(self):
        # Penalty if Y distance between feet grows more than 0.2m
        max_foot_y_distance = self.reward_cfg.get("max_foot_y_distance", 0.2)
        
        # Calculate Y distance between feet
        foot_y_distance = torch.abs(self.left_foot_pos[:, 1] - self.right_foot_pos[:, 1])
        
        # Apply penalty only when distance exceeds threshold
        distance_violation = foot_y_distance > max_foot_y_distance
        penalty = torch.where(
            distance_violation,
            torch.square(foot_y_distance - max_foot_y_distance),  # Quadratic penalty for violation
            torch.zeros_like(foot_y_distance)
        )
        
        return penalty  # Return positive penalty (will be scaled by negative weight)
