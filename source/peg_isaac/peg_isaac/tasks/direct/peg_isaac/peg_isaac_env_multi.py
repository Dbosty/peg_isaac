# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    num_robots = 3
    episode_length_s = 10.0
    decimation = 2
    action_space = 4 * num_robots  # [thrust, moment_x, moment_y, moment_z] for each robot
    observation_space = 12 * num_robots  # [root_lin_vel_b, root_ang_vel_b, projected_gravity_b, desired_pos_b] for each robot
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(
    #     num_envs=4096, env_spacing=2.5, replicate_physics=True, clone_in_fabric=True
    # )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=0.0, replicate_physics=False, clone_in_fabric=False
    )

    # robot
    ########################################################
    robots: list[ArticulationCfg] = [
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0"),
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_1"),
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_2"),
    ]
    ######################################################## 
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, self.cfg.num_robots, 4, device=self.device)
        self._thrust  = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        self._moment  = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
            ]
        }
        # Get specific body indices
        self._body_ids: list[int] = []
        for robot in self._robots:
            self._body_ids.append(robot.find_bodies("body")[0])
        robot0 = self._robots[0]
        self._robot_mass = robot0.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):

        ########################################################
        self._robots: list[Articulation] = []
        

        for i, robot_cfg in enumerate(self.cfg.robots):
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)
            # self._body_ids.append(robot.find_bodies("body")[0])
        ########################################################    

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # we need to explicitly filter collisions for CPU simulation
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)


    def _pre_physics_step(self, actions: torch.Tensor):

        acts = actions.clone().clamp(-1.0, 1.0).view(self.num_envs, self.cfg.num_robots, 4)
        self._actions[:] = acts

        self._thrust.zero_()
        self._thrust[..., 2] = self.cfg.thrust_to_weight * self._robot_weight * (acts[..., 0] + 1.0) / 2.0
        self._moment[:] = self.cfg.moment_scale * acts[..., 1:]

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            robot.set_external_force_and_torque(
                self._thrust[:, i:i+1, :],   # (num_envs, 1, 3)
                self._moment[:, i:i+1, :],   # (num_envs, 1, 3)
                body_ids=self._body_ids[i],
            )

    def _get_observations(self) -> dict:
        obs_per_robot = []
        for i, robot in enumerate(self._robots):
            desired_pos_b, _ = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, self._desired_pos_w[:, i, :]
            )
            obs_i = torch.cat(
                [
                    robot.data.root_lin_vel_b,
                    robot.data.root_ang_vel_b,
                    robot.data.projected_gravity_b,
                    desired_pos_b,
                ],
                dim=-1,
            )  # (num_envs, 12)
            obs_per_robot.append(obs_i)

        obs = torch.stack(obs_per_robot, dim=1)                 # (num_envs, 2, 12)
        return {"policy": obs.reshape(self.num_envs, -1)}       # (num_envs, 24)


    def _get_rewards(self) -> torch.Tensor:
        """Compute reward (summed over the 2 robots) and accumulate episode logs."""
        rewards_lin = torch.zeros(self.num_envs, device=self.device)
        rewards_ang = torch.zeros(self.num_envs, device=self.device)
        rewards_dist = torch.zeros(self.num_envs, device=self.device)

        for i, robot in enumerate(self._robots):
            lin_vel = torch.sum(robot.data.root_lin_vel_b**2, dim=1)
            ang_vel = torch.sum(robot.data.root_ang_vel_b**2, dim=1)
            dist = torch.linalg.norm(self._desired_pos_w[:, i, :] - robot.data.root_pos_w, dim=1)
            dist_mapped = 1.0 - torch.tanh(dist / 0.8)

            rewards_lin = rewards_lin + lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt
            rewards_ang = rewards_ang + ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt
            rewards_dist = rewards_dist + dist_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt

        total = rewards_lin + rewards_ang + rewards_dist

        # Logging (these are episodic accumulators)
        self._episode_sums["lin_vel"] += rewards_lin
        self._episode_sums["ang_vel"] += rewards_ang
        self._episode_sums["distance_to_goal"] += rewards_dist

        return total


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died_any = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for robot in self._robots:
            died = torch.logical_or(robot.data.root_pos_w[:, 2] < 0.1, robot.data.root_pos_w[:, 2] > 2.0)
            died_any |= died
        return died_any, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        """Reset a subset of environments."""
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # ---- logging: final distance-to-goal (mean over robots, mean over env_ids)
        dists = []
        for i, robot in enumerate(self._robots):
            d = torch.linalg.norm(
                self._desired_pos_w[env_ids, i, :] - robot.data.root_pos_w[env_ids],
                dim=1,
            )
            dists.append(d)
        final_distance_to_goal = torch.stack(dists, dim=0).mean()

        extras = {}
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0

        self.extras["log"] = {}
        self.extras["log"].update(extras)

        extras = {}
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # ---- reset bookkeeping (buffers, counters, etc.)
        super()._reset_idx(env_ids)

        # Spread out resets to avoid spikes when resetting all envs
        if len(env_ids) == self.num_envs:
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Clear actions for these envs
        self._actions[env_ids] = 0.0

        # ---- sample new goals (per robot)
        self._desired_pos_w[env_ids, :, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :, :2] += self._terrain.env_origins[env_ids, None, :2]
        self._desired_pos_w[env_ids, :, 2] = torch.zeros_like(self._desired_pos_w[env_ids, :, 2]).uniform_(0.5, 1.5)

        # ---- reset robots (offset so they don't spawn overlapping)
        # grid offsets so robots don't spawn on top of each other
        n = self.cfg.num_robots
        spacing = 0.6

        # make a small grid: (ceil(sqrt(n)) x ceil(sqrt(n)))
        grid = int(torch.ceil(torch.sqrt(torch.tensor(float(n)))).item())
        xy = []
        for r in range(grid):
            for c in range(grid):
                if len(xy) >= n:
                    break
                xy.append([c * spacing, r * spacing, 0.0])
            if len(xy) >= n:
                break

        offsets = torch.tensor(xy, device=self.device)  # (n, 3)

        for i, robot in enumerate(self._robots):
            robot.reset(env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            default_root_state = robot.data.default_root_state[env_ids].clone()

            default_root_state[:, :3] += self._terrain.env_origins[env_ids]
            default_root_state[:, :3] += offsets[i]

            robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        # Visualize robot-0 goal only (since visualize() typically expects (num_envs, 3))
        self.goal_pos_visualizer.visualize(self._desired_pos_w[:, 0, :])
