# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from typing import Dict, Tuple

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_conjugate

from isaaclab_assets import CRAZYFLIE_CFG  # Crazyflie articulation cfg

# Visualization / USD
import carb
import omni.kit.commands
import omni.usd

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from pxr import UsdPhysics, PhysxSchema


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------
@configclass
class PursuitEvasionEnvCfg(DirectMARLEnvCfg):
    # -------------------
    # Multi-agent setup
    # -------------------
    possible_agents = ("pursuer_hi", "pursuer_lo", "evader")

    # Per-agent action/obs spaces (this is what DirectMARLEnvCfg expects)
    action_spaces = {
        "pursuer_hi": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        "pursuer_lo": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        "evader": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
    }

    # pursuer_hi: 12, pursuer_lo: 12, evader: 18
    observation_spaces = {
        "pursuer_hi": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,), dtype=float),
        "pursuer_lo": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,), dtype=float),
        "evader": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(18,), dtype=float),
    }

    # IMPORTANT:
    # IsaacLab hydra serialization expects these fields to NOT be dataclasses.MISSING.
    # Setting them avoids "Unsupported space (MISSING)" errors.
    action_space = gym.spaces.Dict(action_spaces)
    observation_space = gym.spaces.Dict(observation_spaces)
    state_space = gym.spaces.Dict({})

    # -------------------
    # Timing / base env
    # -------------------
    episode_length_s = 20.0
    decimation = 2
    debug_vis = False

    # -------------------
    # Small world (arena)
    # -------------------
    arena_half_extent = 5.0       # x,y in [-5, 5]
    arena_z_min = 0.5
    arena_z_max = 4.0

    wall_height = 3.0
    wall_thickness = 0.2

    enable_camera_follow = True
    camera_offset = (6.0, 6.0, 4.0)  # camera position = evader + offset

    # -------------------
    # Simulation
    # -------------------
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
        physx=sim_utils.PhysxCfg(
            # These help with the GPU broadphase/narrowphase overflow errors you saw
            gpu_found_lost_pairs_capacity=2**23,   # 8,388,608
            gpu_max_rigid_patch_count=2**21,       # 2,097,152
            gpu_max_rigid_contact_count=2**24,     # 16,777,216
        ),
    )

    # -------------------
    # Terrain
    # -------------------
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

    # -------------------
    # Scene
    # -------------------
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=0.0, replicate_physics=False, clone_in_fabric=False
    )

    # -------------------
    # Robots
    # -------------------
    num_robots = 3
    robots: list[ArticulationCfg] = [
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0"),  # pursuer_hi
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_1"),  # pursuer_lo
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_2"),  # evader
    ]

    # -------------------
    # High-level RL command limits (actions are normalized [-1, 1])
    # action = [vx_cmd, vy_cmd, vz_cmd, yaw_rate_cmd] (world frame)
    # -------------------
    v_max_xy = 4.0
    v_max_z = 2.0
    yaw_rate_max = 2.0

    # -------------------
    # Autopilot gains
    # -------------------
    kp_pos = 1.5
    kd_vel = 2.0
    kp_att = 8.0
    kd_omega = 0.25
    tilt_max_rad = 0.6  # ~34 deg

    # -------------------
    # Game parameters
    # -------------------
    capture_radius = 0.6
    goal_radius = 0.6

    # -------------------
    # Observation noise
    # -------------------
    obs_pos_noise_std = 0.05
    obs_vel_noise_std = 0.02


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
class PursuitEvasionDirectMARLEnv(DirectMARLEnv):
    cfg: PursuitEvasionEnvCfg

    PURSUER_HI = 0
    PURSUER_LO = 1
    EVADER = 2

    def __init__(self, cfg: PursuitEvasionEnvCfg, render_mode: str | None = None, **kwargs):
        # NOTE: DirectMARLEnv.__init__ calls _setup_scene() inside.
        super().__init__(cfg, render_mode, **kwargs)

        # Robots are created in _setup_scene()
        # Do NOT assume physx views exist before _setup_scene completes.
        # We only allocate tensors here.

        # High-level commands
        self._v_cmd_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        self._yaw_cmd = torch.zeros(self.num_envs, self.cfg.num_robots, device=self.device)
        self._pos_cmd_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)

        # Wrench buffers (world)
        self._force_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        self._torque_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)

        # Evader goal
        self._goal_evader_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Constants
        self._g_w = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float).view(1, 3)
        self._e3_w = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float).view(1, 3)

        # Will be filled in _setup_scene
        # self._robots: list[Articulation] = []
        self._body_ids: list[int] = []
        # self._robot_mass: torch.Tensor | None = None
        # Now PhysX views exist -> safe to query bodies/mass
        self._body_ids = [r.find_bodies("body")[0] for r in self._robots]
        self._robot_mass = self._robots[0].root_physx_view.get_masses()[0].sum()

    # --------------------------
    # Scene setup
    # --------------------------
    def _setup_scene(self):
        # Create robots and register them to the scene
        self._robots: list[Articulation] = []
        # self._body_ids = []

        for i, robot_cfg in enumerate(self.cfg.robots):
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)

        # Terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone envs
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(1.0, 1.0, 1.0))
        light_cfg.func("/World/Light", light_cfg)

        

        # World dressing / visuals
        self._set_white_background()
        self._create_arena_walls()
        self._create_capture_visual()
        self._create_follow_camera()
        self._color_code_drones()

    # --------------------------
    # Step callbacks
    # --------------------------
    def _pre_physics_step(self, actions: Dict[str, torch.Tensor]):
        # actions: dict(agent)->(N,4)
        a_hi = actions["pursuer_hi"].clamp(-1.0, 1.0)
        a_lo = actions["pursuer_lo"].clamp(-1.0, 1.0)
        a_ev = actions["evader"].clamp(-1.0, 1.0)

        A = torch.stack([a_hi, a_lo, a_ev], dim=1)  # (N,3,4)

        # velocity commands (world)
        self._v_cmd_w[..., 0:2] = A[..., 0:2] * self.cfg.v_max_xy
        self._v_cmd_w[..., 2] = A[..., 2] * self.cfg.v_max_z

        # yaw integration from yaw-rate
        yaw_rate_cmd = A[..., 3] * self.cfg.yaw_rate_max
        self._yaw_cmd = self._yaw_cmd + yaw_rate_cmd * self.step_dt

        # pos anchor integrates velocity (smooth)
        self._pos_cmd_w = self._pos_cmd_w + self._v_cmd_w * self.step_dt

        # wrench from autopilot
        self._compute_autopilot_wrench()

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            force = self._force_w[:, i:i + 1, :]   # (N,1,3)
            torque = self._torque_w[:, i:i + 1, :] # (N,1,3)

            if hasattr(robot, "set_forces_and_torques"):
                robot.set_forces_and_torques(force, torque, body_ids=self._body_ids[i])
            else:
                robot.set_external_force_and_torque(force, torque, body_ids=self._body_ids[i])

    def _post_physics_step(self):
        super()._post_physics_step()
        self._update_capture_visual()
        self._update_follow_camera()

    # --------------------------
    # Autopilot: v_cmd -> wrench
    # --------------------------
    def _compute_autopilot_wrench(self):
        self._force_w.zero_()
        self._torque_w.zero_()

        # lateral accel clamp from tilt limit
        a_lat_max = torch.tan(torch.tensor(self.cfg.tilt_max_rad, device=self.device)) * 9.81

        for i, robot in enumerate(self._robots):
            pos_w = robot.data.root_pos_w
            vel_w = robot.data.root_lin_vel_w
            quat_w = robot.data.root_quat_w
            omg_b = robot.data.root_ang_vel_b

            p_cmd = self._pos_cmd_w[:, i, :]
            v_cmd = self._v_cmd_w[:, i, :]
            yaw_cmd = self._yaw_cmd[:, i]

            # outer loop accel
            a_des = self.cfg.kp_pos * (p_cmd - pos_w) + self.cfg.kd_vel * (v_cmd - vel_w)

            # clamp lateral accel
            a_lat = a_des.clone()
            a_lat[:, 2] = 0.0
            n = torch.linalg.norm(a_lat, dim=1).clamp(min=1e-6)
            scale = torch.minimum(torch.ones_like(n), a_lat_max / n)
            a_des[:, 0] *= scale
            a_des[:, 1] *= scale

            # gravity compensation
            a_total = a_des - self._g_w  # since gravity is negative z

            # desired body-z direction
            b3_des = a_total / torch.linalg.norm(a_total, dim=1, keepdim=True).clamp(min=1e-6)

            # current body-z in world
            b3_cur = quat_apply(quat_w, self._e3_w.expand_as(pos_w))

            # desired thrust along current b3 axis
            F_w = self._robot_mass * a_total
            thrust_mag = torch.sum(F_w * b3_cur, dim=1).clamp(min=0.0)
            self._force_w[:, i, :] = thrust_mag.unsqueeze(-1) * b3_cur

            # yaw frame
            c = torch.cos(yaw_cmd)
            s = torch.sin(yaw_cmd)
            b1_yaw = torch.stack([c, s, torch.zeros_like(c)], dim=1)

            b2_des = torch.cross(b3_des, b1_yaw, dim=1)
            b2_des = b2_des / torch.linalg.norm(b2_des, dim=1, keepdim=True).clamp(min=1e-6)
            b1_des = torch.cross(b2_des, b3_des, dim=1)

            # simple attitude error: align b3
            e_R_w = torch.cross(b3_cur, b3_des, dim=1)
            e_R_b = quat_apply(quat_conjugate(quat_w), e_R_w)

            torque_b = (-self.cfg.kp_att * e_R_b) + (-self.cfg.kd_omega * omg_b)
            self._torque_w[:, i, :] = quat_apply(quat_w, torque_b)

    # --------------------------
    # Observations
    # IMPORTANT: return dict[str, torch.Tensor] (not nested dict) for skrl wrapper
    # --------------------------
    def _get_observations(self) -> Dict[str, torch.Tensor]:
        r_hi = self._robots[self.PURSUER_HI]
        r_lo = self._robots[self.PURSUER_LO]
        r_ev = self._robots[self.EVADER]

        def self_state(r: Articulation) -> torch.Tensor:
            return torch.cat(
                [r.data.root_lin_vel_b, r.data.root_ang_vel_b, r.data.projected_gravity_b],
                dim=-1,
            )  # (N,9)

        rel_hi_to_ev = (r_ev.data.root_pos_w - r_hi.data.root_pos_w)
        rel_lo_to_ev = (r_ev.data.root_pos_w - r_lo.data.root_pos_w)

        rel_ev_to_goal = (self._goal_evader_w - r_ev.data.root_pos_w)
        rel_ev_to_hi = (r_hi.data.root_pos_w - r_ev.data.root_pos_w)
        rel_ev_to_lo = (r_lo.data.root_pos_w - r_ev.data.root_pos_w)

        if self.cfg.obs_pos_noise_std > 0:
            n = self.cfg.obs_pos_noise_std
            rel_hi_to_ev = rel_hi_to_ev + n * torch.randn_like(rel_hi_to_ev)
            rel_lo_to_ev = rel_lo_to_ev + n * torch.randn_like(rel_lo_to_ev)
            rel_ev_to_goal = rel_ev_to_goal + n * torch.randn_like(rel_ev_to_goal)
            rel_ev_to_hi = rel_ev_to_hi + n * torch.randn_like(rel_ev_to_hi)
            rel_ev_to_lo = rel_ev_to_lo + n * torch.randn_like(rel_ev_to_lo)

        obs_hi = torch.cat([self_state(r_hi), rel_hi_to_ev], dim=-1)  # (N,12)
        obs_lo = torch.cat([self_state(r_lo), rel_lo_to_ev], dim=-1)  # (N,12)
        obs_ev = torch.cat([self_state(r_ev), rel_ev_to_goal, rel_ev_to_hi, rel_ev_to_lo], dim=-1)  # (N,18)

        return {"pursuer_hi": obs_hi, "pursuer_lo": obs_lo, "evader": obs_ev}

    # --------------------------
    # Rewards
    # --------------------------
    def _get_rewards(self) -> Dict[str, torch.Tensor]:
        p_hi = self._robots[self.PURSUER_HI].data.root_pos_w
        p_lo = self._robots[self.PURSUER_LO].data.root_pos_w
        p_ev = self._robots[self.EVADER].data.root_pos_w

        dist_lo_ev = torch.linalg.norm(p_ev - p_lo, dim=1)
        dist_hi_ev = torch.linalg.norm(p_ev - p_hi, dim=1)
        dist_ev_goal = torch.linalg.norm(self._goal_evader_w - p_ev, dim=1)

        capture = dist_lo_ev < self.cfg.capture_radius
        success = dist_ev_goal < self.cfg.goal_radius

        r_pursuer = -0.2 * dist_lo_ev - 0.05 * dist_hi_ev + 10.0 * capture.float()

        threat = 1.0 / (dist_lo_ev + 1e-3) + 0.5 / (dist_hi_ev + 1e-3)
        r_evader = -0.3 * dist_ev_goal - 0.2 * threat + 10.0 * success.float() - 10.0 * capture.float()

        return {"pursuer_hi": r_pursuer, "pursuer_lo": r_pursuer, "evader": r_evader}

    # --------------------------
    # Dones
    # --------------------------
    def _get_dones(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        p_lo = self._robots[self.PURSUER_LO].data.root_pos_w
        p_ev = self._robots[self.EVADER].data.root_pos_w

        dist_lo_ev = torch.linalg.norm(p_ev - p_lo, dim=1)
        dist_ev_goal = torch.linalg.norm(self._goal_evader_w - p_ev, dim=1)

        capture = dist_lo_ev < self.cfg.capture_radius
        success = dist_ev_goal < self.cfg.goal_radius

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = capture | success
        truncated = time_out

        term = {a: terminated for a in self.cfg.possible_agents}
        trunc = {a: truncated for a in self.cfg.possible_agents}
        return term, trunc

    # --------------------------
    # Reset
    # --------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        super()._reset_idx(env_ids)

        origin = self._terrain.env_origins[env_ids]  # (E,3)

        spawn_offsets = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.5, 0.0, 1.0],
                [-0.5, 0.0, 1.0],
            ],
            device=self.device,
            dtype=torch.float,
        )

        # spawn_offsets = torch.tensor(
        #     [
        #         [0.0, 0.0, 3.0],    # pursuer_hi
        #         [2.0, 0.0, 2.0],    # pursuer_lo
        #         [-2.0, 0.0, 2.0],   # evader
        #     ],
        #     device=self.device,
        #     dtype=torch.float,
        # )

        for i, robot in enumerate(self._robots):
            robot.reset(env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            root_state = robot.data.default_root_state[env_ids].clone()

            print(f'Resetting Robot {i} at:\n{origin + spawn_offsets[i]}')

            root_state[:, :3] = origin + spawn_offsets[i]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        print("Robot world positions:")
        for i in range(3):
            print(self._robots[i].data.root_pos_w[0])

        print("===============================\n")

        # initialize command anchors to current states
        for i, robot in enumerate(self._robots):
            self._pos_cmd_w[env_ids, i, :] = robot.data.root_pos_w[env_ids]
            self._yaw_cmd[env_ids, i] = 0.0
            self._v_cmd_w[env_ids, i, :] = 0.0

        # sample a new evader goal within arena
        half = float(self.cfg.arena_half_extent)
        goal_xy = torch.empty((len(env_ids), 2), device=self.device).uniform_(-half, half)
        self._goal_evader_w[env_ids, 0:2] = origin[:, 0:2] + goal_xy
        self._goal_evader_w[env_ids, 2] = 1.0

    # ----------------------------------------------------------------------------------
    # VISUALS / WORLD DRESSING
    # ----------------------------------------------------------------------------------
    def _get_stage(self) -> Usd.Stage:
        # robust stage access
        ctx = omni.usd.get_context()
        stg = ctx.get_stage()
        if stg is None:
            # fallback
            stg = sim_utils.SimulationContext.instance().stage
        return stg

    def _set_white_background(self):
        try:
            settings = carb.settings.get_settings()
            settings.set("/app/renderer/clearColor", [1.0, 1.0, 1.0, 1.0])
            settings.set("/rtx/sky/enabled", False)
        except Exception as e:
            print("White background setup skipped:", e)

    def _make_preview_material(self, stage: Usd.Stage, material_path: str, rgb: tuple[float, float, float]):
        prim = stage.GetPrimAtPath(material_path)
        if prim.IsValid():
            return UsdShade.Material(prim)

        mat = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat_out = mat.CreateSurfaceOutput()
        mat_out.ConnectToSource(shader_out)

        return mat

    def _bind_material_recursive(self, stage: Usd.Stage, root_prim_path: str, material: UsdShade.Material):
        root = stage.GetPrimAtPath(root_prim_path)
        if not root.IsValid():
            return

        def bind(prim):
            try:
                UsdShade.MaterialBindingAPI(prim).Bind(material)
            except Exception:
                pass

        bind(root)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                bind(prim)

    def _find_robot_root_prim(self, robot_index: int) -> str | None:
        """Find a prim path ending with '/Robot_{i}' under /World/envs for env_0.
        Avoid hardcoding /World/envs/env_0 because different Isaac versions may vary."""
        stage = self._get_stage()
        target_suffix = f"/Robot_{robot_index}"

        # Fast path guesses
        candidates = [
            f"/World/envs/env_0/Robot_{robot_index}",
            f"/World/envs/envs_0/Robot_{robot_index}",
        ]
        for c in candidates:
            if stage.GetPrimAtPath(c).IsValid():
                return c

        # Search under /World/envs
        envs = stage.GetPrimAtPath("/World/envs")
        if not envs.IsValid():
            return None

        for prim in Usd.PrimRange(envs):
            p = prim.GetPath().pathString
            if p.endswith(target_suffix) and ("/env_0/" in p or "/envs_0/" in p or "/env0/" in p):
                return p

        # If still not found, return any suffix match
        for prim in Usd.PrimRange(envs):
            p = prim.GetPath().pathString
            if p.endswith(target_suffix):
                return p

        return None

    def _create_wall(self, stage: Usd.Stage, path: str, center: tuple[float, float, float], size: tuple[float, float, float]):
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return

        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(1.0)

        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))
        xform.AddScaleOp().Set(Gf.Vec3f(*size))

        # collision
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())

    def _create_arena_walls(self):
        stage = self._get_stage()

        half = float(self.cfg.arena_half_extent)
        h = float(self.cfg.wall_height)
        t = float(self.cfg.wall_thickness)
        zc = h * 0.5

        self._create_wall(stage, "/World/Obstacles/Wall_PosX", (half, 0.0, zc), (t, 2 * half + t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_NegX", (-half, 0.0, zc), (t, 2 * half + t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_PosY", (0.0, half, zc), (2 * half + t, t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_NegY", (0.0, -half, zc), (2 * half + t, t, h))

    def _create_capture_visual(self):
        stage = self._get_stage()
        path = "/World/Visuals/CaptureRadius"
        if stage.GetPrimAtPath(path).IsValid():
            return

        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(self.cfg.capture_radius))

        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        mat = self._make_preview_material(stage, "/World/Materials/CaptureBlue", (0.2, 0.4, 1.0))
        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)

    def _update_capture_visual(self):
        stage = self._get_stage()
        prim = stage.GetPrimAtPath("/World/Visuals/CaptureRadius")
        if not prim.IsValid():
            return

        ev = self._robots[self.EVADER].data.root_pos_w
        p = ev[0].tolist()

        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        if not ops:
            xform.AddTranslateOp().Set(Gf.Vec3d(*p))
        else:
            ops[0].Set(Gf.Vec3d(*p))

    def _create_follow_camera(self):
        stage = self._get_stage()
        cam_path = "/World/CameraFollow"
        if stage.GetPrimAtPath(cam_path).IsValid():
            return
        UsdGeom.Camera.Define(stage, cam_path)

    def _update_follow_camera(self):
        if not self.cfg.enable_camera_follow:
            return

        stage = self._get_stage()
        cam_path = "/World/CameraFollow"
        cam = stage.GetPrimAtPath(cam_path)
        if not cam.IsValid():
            return

        ev_pos = self._robots[self.EVADER].data.root_pos_w[0]
        off = torch.tensor(self.cfg.camera_offset, device=self.device, dtype=torch.float)
        cam_pos = ev_pos + off

        xform = UsdGeom.Xformable(cam)
        ops = xform.GetOrderedXformOps()
        if len(ops) == 0:
            t_op = xform.AddTranslateOp()
            r_op = xform.AddRotateXYZOp()
        else:
            t_op = ops[0]
            r_op = ops[1] if len(ops) > 1 else xform.AddRotateXYZOp()

        t_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))

        look = (ev_pos - cam_pos)
        yaw = math.degrees(math.atan2(float(look[1]), float(look[0])))
        dist_xy = math.sqrt(float(look[0] ** 2 + look[1] ** 2)) + 1e-6
        pitch = -math.degrees(math.atan2(float(look[2]), dist_xy))

        r_op.Set(Gf.Vec3f(float(pitch), 0.0, float(yaw)))

        try:
            omni.kit.commands.execute("SetViewportCamera", camera_path=cam_path)
        except Exception:
            pass

    def _color_code_drones(self):
        stage = self._get_stage()

        mat_red = self._make_preview_material(stage, "/World/Materials/PursuerHiRed", (1.0, 0.2, 0.2))
        mat_grn = self._make_preview_material(stage, "/World/Materials/PursuerLoGreen", (0.2, 1.0, 0.2))
        mat_blk = self._make_preview_material(stage, "/World/Materials/EvaderBlack", (0.05, 0.05, 0.05))

        p0 = self._find_robot_root_prim(0)
        p1 = self._find_robot_root_prim(1)
        p2 = self._find_robot_root_prim(2)

        if p0:
            self._bind_material_recursive(stage, p0, mat_red)
        if p1:
            self._bind_material_recursive(stage, p1, mat_grn)
        if p2:
            self._bind_material_recursive(stage, p2, mat_blk)
