# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
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

from isaaclab_assets import CRAZYFLIE_CFG  # your crazyflie config

# For visualizations
import carb
import omni.kit.commands

from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf
from pxr import UsdPhysics, PhysxSchema



# -------------------------
# Config
# -------------------------
@configclass
class PursuitEvasionEnvCfg(DirectMARLEnvCfg):
    ########################################################
    #                       Agents                         
    ########################################################
    possible_agents = ("pursuer_hi", "pursuer_lo", "evader")

    ########################################################
    #                   State Spaces                         
    ########################################################
    # --- agent action/observation spaces for DirectMARLEnvCfg
    action_spaces = {
        "pursuer_hi": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        "pursuer_lo": gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
        "evader":     gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=float),
    }

    # Your env currently returns obs sizes:
    # pursuer_hi: 12, pursuer_lo: 12, evader: 18
    observation_spaces = {
        "pursuer_hi": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,), dtype=float),
        "pursuer_lo": gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(12,), dtype=float),
        "evader":     gym.spaces.Box(low=-float("inf"), high=float("inf"), shape=(18,), dtype=float),
    }

    # --- ALSO set the singular spaces (what Hydra is probably serializing)
    # Actions passed to env.step are a dict keyed by agent -> (N,4)
    action_space = gym.spaces.Dict(action_spaces)

    # Observations returned are dict keyed by agent -> {"policy": obs}
    # So the observation space should reflect that nested dict structure.
    observation_space = gym.spaces.Dict(observation_spaces)

    # Some cfgs also include state_space; set it to an empty Dict to avoid MISSING
    state_space = gym.spaces.Dict({})

    ########################################################
    #                       Environment                        
    ########################################################
    # env timing
    episode_length_s = 20.0
    decimation = 2
    debug_vis = False

    # Arena bounds
    arena_half_extent = 5.0       # small world: x,y in [-5, 5]
    arena_z_min = 0.5
    arena_z_max = 4.0

    # Wall geometry
    wall_height = 3.0
    wall_thickness = 0.2

    # Visuals
    enable_camera_follow = True
    camera_offset = (6.0, 6.0, 4.0)     # world offset from evader
    capture_radius = 0.6                # you already have this


    # --- simulation
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
        # Your error wants ~4,498,500 pairs -> set to next power of 2
        gpu_found_lost_pairs_capacity=2**23,      # 8,388,608

        # Your error wants patch buffer >= ~2,097,152 -> set to at least this
        gpu_max_rigid_patch_count=2**21,          # 2,097,152  (try 2**22 if still overflows)

        # Often helps when patch count explodes
        gpu_max_rigid_contact_count=2**24,  
        )
    )

    # --- terrain (plane for now)
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

    # --- scene (start with 1 env for debugging; scale later)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=1, env_spacing=0.0, replicate_physics=False, clone_in_fabric=False
    )

    # --- robots
    num_robots = 3
    robots: list[ArticulationCfg] = [
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_0"),  # pursuer_hi
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_1"),  # pursuer_lo
        CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot_2"),  # evader
    ]

    # --- high-level action limits (RL outputs normalized [-1,1])
    v_max_xy = 4.0         # m/s
    v_max_z = 2.0          # m/s
    yaw_rate_max = 2.0     # rad/s

    # --- autopilot gains (tune later)
    kp_pos = 1.5
    kd_vel = 2.0
    kp_att = 8.0
    kd_omega = 0.25
    tilt_max_rad = 0.6     # ~34 deg

    # --- game parameters
    capture_radius = 0.6   # meters (low pursuer catches evader)
    goal_radius = 0.6      # meters (evader wins)
    arena_half_extent = 10.0  # clamp/play area for goal sampling (x,y)

    # --- observation noise (simple)
    obs_pos_noise_std = 0.05   # meters (relative vectors)
    obs_vel_noise_std = 0.02   # m/s


# -------------------------
# Environment
# -------------------------
class PursuitEvasionDirectMARLEnv(DirectMARLEnv):
    cfg: PursuitEvasionEnvCfg

    # fixed role->index mapping
    PURSUER_HI = 0
    PURSUER_LO = 1
    EVADER = 2

    def __init__(self, cfg: PursuitEvasionEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # # Robot handles
        # self._robots: list[Articulation] = []
        # self._body_ids: list[int] = []

        # Command buffers (high-level)
        self._v_cmd_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        self._yaw_cmd = torch.zeros(self.num_envs, self.cfg.num_robots, device=self.device)

        # Position hold anchor (helps make velocity commands fly nicely)
        self._pos_cmd_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)

        # Wrench buffers
        self._force_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)
        self._torque_w = torch.zeros(self.num_envs, self.cfg.num_robots, 3, device=self.device)

        # Game state
        self._goal_evader_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Constants
        self._g_w = torch.tensor(self.sim.cfg.gravity, device=self.device, dtype=torch.float).view(1, 3)
        self._e3_w = torch.tensor([0.0, 0.0, 1.0], device=self.device, dtype=torch.float).view(1, 3)

        # Mass
        self._body_ids: list[int] = []
        for robot in self._robots:
            self._body_ids.append(robot.find_bodies("body")[0])
        robot0 = self._robots[0]
        self._robot_mass = robot0.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()
        self._robot_mass = self._robots[0].root_physx_view.get_masses()[0].sum()

    # ---- scene setup
    def _setup_scene(self):
        # self._robots = []

        # for i, robot_cfg in enumerate(self.cfg.robots):
        #     robot = Articulation(robot_cfg)
        #     self.scene.articulations[f"robot_{i}"] = robot
        #     self._robots.append(robot)
        
        self._robots: list[Articulation] = []
        
        for i, robot_cfg in enumerate(self.cfg.robots):
            robot = Articulation(robot_cfg)
            self.scene.articulations[f"robot_{i}"] = robot
            self._robots.append(robot)

        # terrain
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # clone envs
        self.scene.clone_environments(copy_from_source=False)

        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # light
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # visuals / world dressing
        self._set_white_background()
        self._create_arena_walls()
        self._create_capture_visual()
        self._create_follow_camera()

        # --- recolor robots (safe to do after clone_environments)
        self._color_code_drones()


    # ---- actions: dict(agent)->(N,4)
    def _pre_physics_step(self, actions: dict[str, torch.Tensor]):
        # stack agent actions into robot order: [hi, lo, ev]
        a_hi = actions["pursuer_hi"].clamp(-1.0, 1.0)
        a_lo = actions["pursuer_lo"].clamp(-1.0, 1.0)
        a_ev = actions["evader"].clamp(-1.0, 1.0)

        A = torch.stack([a_hi, a_lo, a_ev], dim=1)  # (N,3,4)

        # velocity commands (world)
        self._v_cmd_w[..., 0:2] = A[..., 0:2] * self.cfg.v_max_xy
        self._v_cmd_w[..., 2] = A[..., 2] * self.cfg.v_max_z

        # integrate yaw command from yaw-rate
        yaw_rate_cmd = A[..., 3] * self.cfg.yaw_rate_max
        self._yaw_cmd = self._yaw_cmd + yaw_rate_cmd * self.step_dt

        # position anchor update (turns vel commands into smooth motion)
        # if you want pure vel tracking, set kp_pos=0 in cfg and this becomes irrelevant.
        self._pos_cmd_w = self._pos_cmd_w + self._v_cmd_w * self.step_dt

        # compute wrench
        self._compute_autopilot_wrench()

    def _compute_autopilot_wrench(self):
        self._force_w.zero_()
        self._torque_w.zero_()

        a_lat_max = torch.tan(torch.tensor(self.cfg.tilt_max_rad, device=self.device)) * 9.81

        for i, robot in enumerate(self._robots):
            pos_w = robot.data.root_pos_w          # (N,3)
            vel_w = robot.data.root_lin_vel_w      # (N,3)
            quat_w = robot.data.root_quat_w        # (N,4)
            omg_b = robot.data.root_ang_vel_b      # (N,3)

            p_cmd = self._pos_cmd_w[:, i, :]
            v_cmd = self._v_cmd_w[:, i, :]
            yaw_cmd = self._yaw_cmd[:, i]

            # outer-loop accel command
            a_des = self.cfg.kp_pos * (p_cmd - pos_w) + self.cfg.kd_vel * (v_cmd - vel_w)

            # clamp lateral accel
            a_lat = a_des.clone()
            a_lat[:, 2] = 0.0
            n = torch.linalg.norm(a_lat, dim=1).clamp(min=1e-6)
            scale = torch.minimum(torch.ones_like(n), a_lat_max / n)
            a_des[:, 0] *= scale
            a_des[:, 1] *= scale

            # gravity compensation
            a_total = a_des - self._g_w

            # desired thrust direction
            b3_des = a_total / torch.linalg.norm(a_total, dim=1, keepdim=True).clamp(min=1e-6)

            # current body z axis in world
            b3_cur = quat_apply(quat_w, self._e3_w.expand_as(pos_w))

            # force vector (world)
            F_w = self._robot_mass * a_total
            thrust_mag = torch.sum(F_w * b3_cur, dim=1).clamp(min=0.0)
            force_w = thrust_mag.unsqueeze(-1) * b3_cur
            self._force_w[:, i, :] = force_w

            # attitude error (simple): align z axis, plus yaw heading
            c = torch.cos(yaw_cmd)
            s = torch.sin(yaw_cmd)
            b1_yaw = torch.stack([c, s, torch.zeros_like(c)], dim=1)

            b2_des = torch.cross(b3_des, b1_yaw, dim=1)
            b2_des = b2_des / torch.linalg.norm(b2_des, dim=1, keepdim=True).clamp(min=1e-6)
            b1_des = torch.cross(b2_des, b3_des, dim=1)

            # z-axis alignment error
            e_R_w = torch.cross(b3_cur, b3_des, dim=1)
            e_R_b = quat_apply(quat_conjugate(quat_w), e_R_w)

            # body torque -> world torque
            torque_b = (-self.cfg.kp_att * e_R_b) + (-self.cfg.kd_omega * omg_b)
            torque_w = quat_apply(quat_w, torque_b)

            self._torque_w[:, i, :] = torque_w

    def _apply_action(self):
        for i, robot in enumerate(self._robots):
            force = self._force_w[:, i:i+1, :]   # (N,1,3)
            torque = self._torque_w[:, i:i+1, :] # (N,1,3)
            # Newer API (recommended). If your install lacks it, fall back to set_external_force_and_torque.
            if hasattr(robot, "set_forces_and_torques"):
                robot.set_forces_and_torques(force, torque, body_ids=self._body_ids[i])
            else:
                robot.set_external_force_and_torque(force, torque, body_ids=self._body_ids[i])
        
        self._update_capture_visual()
        self._update_follow_camera()


    # ---- observations (partial)
    def _get_observations(self) -> dict[str, torch.Tensor]:
        r_hi = self._robots[self.PURSUER_HI]
        r_lo = self._robots[self.PURSUER_LO]
        r_ev = self._robots[self.EVADER]

        # self state (body-frame)
        def self_state(r: Articulation):
            return torch.cat(
                [
                    r.data.root_lin_vel_b,          # (N,3)
                    r.data.root_ang_vel_b,          # (N,3)
                    r.data.projected_gravity_b,     # (N,3)
                ],
                dim=-1,
            )  # (N,9)

        # relative vectors in world
        rel_hi_to_ev = (r_ev.data.root_pos_w - r_hi.data.root_pos_w)  # (N,3)
        rel_lo_to_ev = (r_ev.data.root_pos_w - r_lo.data.root_pos_w)  # (N,3)
        rel_ev_to_goal = (self._goal_evader_w - r_ev.data.root_pos_w) # (N,3)
        rel_ev_to_hi = (r_hi.data.root_pos_w - r_ev.data.root_pos_w)  # (N,3)
        rel_ev_to_lo = (r_lo.data.root_pos_w - r_ev.data.root_pos_w)  # (N,3)

        # add simple Gaussian noise to relative position signals
        if self.cfg.obs_pos_noise_std > 0:
            n = self.cfg.obs_pos_noise_std
            rel_hi_to_ev = rel_hi_to_ev + n * torch.randn_like(rel_hi_to_ev)
            rel_lo_to_ev = rel_lo_to_ev + n * torch.randn_like(rel_lo_to_ev)
            rel_ev_to_goal = rel_ev_to_goal + n * torch.randn_like(rel_ev_to_goal)
            rel_ev_to_hi = rel_ev_to_hi + n * torch.randn_like(rel_ev_to_hi)
            rel_ev_to_lo = rel_ev_to_lo + n * torch.randn_like(rel_ev_to_lo)

        # build per-agent obs
        # pursuer_hi: self + relative-to-evader + (optional) relative-to-evader-velocity
        obs_hi = torch.cat([self_state(r_hi), rel_hi_to_ev], dim=-1)  # (N,12)

        # pursuer_lo: self + relative-to-evader
        obs_lo = torch.cat([self_state(r_lo), rel_lo_to_ev], dim=-1)  # (N,12)

        # evader: self + relative-to-goal + relative-to-both pursuers
        obs_ev = torch.cat([self_state(r_ev), rel_ev_to_goal, rel_ev_to_hi, rel_ev_to_lo], dim=-1)  # (N,18)

        return {
            "pursuer_hi": obs_hi,
            "pursuer_lo": obs_lo,
            "evader": obs_ev,
        }

    # ---- rewards
    def _get_rewards(self) -> dict[str, torch.Tensor]:
        p_hi = self._robots[self.PURSUER_HI].data.root_pos_w
        p_lo = self._robots[self.PURSUER_LO].data.root_pos_w
        p_ev = self._robots[self.EVADER].data.root_pos_w

        dist_lo_ev = torch.linalg.norm(p_ev - p_lo, dim=1)
        dist_hi_ev = torch.linalg.norm(p_ev - p_hi, dim=1)
        dist_ev_goal = torch.linalg.norm(self._goal_evader_w - p_ev, dim=1)

        capture = dist_lo_ev < self.cfg.capture_radius
        success = dist_ev_goal < self.cfg.goal_radius

        # pursuers: move toward evader + capture bonus
        r_pursuer = -0.2 * dist_lo_ev - 0.05 * dist_hi_ev
        r_pursuer = r_pursuer + 10.0 * capture.float()

        # evader: move toward goal + avoid pursuers + success bonus + capture penalty
        threat = 1.0 / (dist_lo_ev + 1e-3) + 0.5 / (dist_hi_ev + 1e-3)
        r_evader = -0.3 * dist_ev_goal - 0.2 * threat
        r_evader = r_evader + 10.0 * success.float() - 10.0 * capture.float()

        return {
            "pursuer_hi": r_pursuer,
            "pursuer_lo": r_pursuer,
            "evader": r_evader,
        }

    # ---- terminations
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        p_lo = self._robots[self.PURSUER_LO].data.root_pos_w
        p_ev = self._robots[self.EVADER].data.root_pos_w

        dist_lo_ev = torch.linalg.norm(p_ev - p_lo, dim=1)
        dist_ev_goal = torch.linalg.norm(self._goal_evader_w - p_ev, dim=1)

        capture = dist_lo_ev < self.cfg.capture_radius
        success = dist_ev_goal < self.cfg.goal_radius

        # timeout from base buffers
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        terminated = capture | success
        truncated = time_out

        term = {
            "pursuer_hi": terminated,
            "pursuer_lo": terminated,
            "evader": terminated,
        }
        trunc = {
            "pursuer_hi": truncated,
            "pursuer_lo": truncated,
            "evader": truncated,
        }
        return term, trunc

    # ---- reset
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robots[0]._ALL_INDICES

        # reset base bookkeeping
        super()._reset_idx(env_ids)

        # reset robots (spread spawns)
        origin = self._terrain.env_origins[env_ids]  # (E,3)

        # fixed role altitudes (overwatch high; low pursuer + evader lower)
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

        # spawn_offsets = torch.tensor(
        #     [
        #         [0.0, 0.0, 2.5],   # pursuer_hi
        #         [1.0, 0.0, 1.0],   # pursuer_lo
        #         [-1.0, 0.0, 1.0],  # evader
        #     ],
        #     device=self.device,
        #     dtype=torch.float,
        # )  # (3,3)

        for i, robot in enumerate(self._robots):
            robot.reset(env_ids)

            joint_pos = robot.data.default_joint_pos[env_ids]
            joint_vel = robot.data.default_joint_vel[env_ids]
            root_state = robot.data.default_root_state[env_ids].clone()

            root_state[:, :3] = origin + spawn_offsets[i]
            robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
            robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
            robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # initialize command anchors to current positions so we don't "jump"
        for i, robot in enumerate(self._robots):
            self._pos_cmd_w[env_ids, i, :] = robot.data.root_pos_w[env_ids]
            self._yaw_cmd[env_ids, i] = 0.0
            self._v_cmd_w[env_ids, i, :] = 0.0

        # sample a new evader goal (x,y) within arena bounds, z around 1.0
        half = self.cfg.arena_half_extent
        goal_xy = torch.empty((len(env_ids), 2), device=self.device).uniform_(-half, half)
        self._goal_evader_w[env_ids, 0:2] = origin[:, 0:2] + goal_xy
        self._goal_evader_w[env_ids, 2] = 1.0


    # ---------------------------------
    # White background + white lighting
    # ---------------------------------
    def _set_white_background(self):
        # Makes viewport clear color white
        try:
            settings = carb.settings.get_settings()
            settings.set("/app/renderer/clearColor", [1.0, 1.0, 1.0, 1.0])
            # Optional: disable sky if present
            settings.set("/rtx/sky/enabled", False)
        except Exception as e:
            print("White background setup skipped:", e)

    # ---------------------------------
    # Material helpers (for ground + drone colors)
    # ---------------------------------
    def _make_preview_material(self, stage, material_path: str, rgb: tuple[float, float, float]):
        # If already exists, return it
        prim = stage.GetPrimAtPath(material_path)
        if prim.IsValid():
            return UsdShade.Material(prim)

        # Material + shader
        mat = UsdShade.Material.Define(stage, material_path)
        shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
        shader.CreateIdAttr("UsdPreviewSurface")

        # Inputs
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*rgb))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

        # OUTPUTS: create shader output and connect material surface to it
        shader_out = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        mat_out = mat.CreateSurfaceOutput()
        mat_out.ConnectToSource(shader_out)

        return mat

    def _bind_material_recursive(self, stage: Usd.Stage, root_prim_path: str, material: UsdShade.Material):
        root = stage.GetPrimAtPath(root_prim_path)
        if not root.IsValid():
            print("Material bind: prim not found:", root_prim_path)
            return

        # Bind to root and all Mesh descendants
        def bind(prim):
            UsdShade.MaterialBindingAPI(prim).Bind(material)

        bind(root)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                bind(prim)

    # ---------------------------------
    # Arena walls (visual + collision)
    # ---------------------------------
    def _create_wall(self, stage: Usd.Stage, path: str, center: tuple[float, float, float], size: tuple[float, float, float]):
        # Creates a cube scaled to size, with collision enabled
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return

        cube = UsdGeom.Cube.Define(stage, path)
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*center))

        # Cube size is side length; we use scale to set dimensions
        # Default cube is 2 units in some contexts; safest: set size=1 then scale to desired dims
        cube.CreateSizeAttr(1.0)
        xform.AddScaleOp().Set(Gf.Vec3f(*size))

        # Enable collision
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())

    def _create_arena_walls(self):
        stage = sim_utils.SimulationContext.instance().stage

        half = float(self.cfg.arena_half_extent)
        h = float(self.cfg.wall_height)
        t = float(self.cfg.wall_thickness)

        # Walls are centered around origin; if you use env_origins later, we can offset them per-env.
        # For num_envs=1 this is perfect.
        zc = h * 0.5

        # Dimensions: (sx, sy, sz) as scale (not meters exactly, but proportional). We'll treat as meters.
        # Place 4 walls: +X, -X, +Y, -Y
        self._create_wall(stage, "/World/Obstacles/Wall_PosX", ( half, 0.0, zc), (t, 2*half + t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_NegX", (-half, 0.0, zc), (t, 2*half + t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_PosY", (0.0,  half, zc), (2*half + t, t, h))
        self._create_wall(stage, "/World/Obstacles/Wall_NegY", (0.0, -half, zc), (2*half + t, t, h))

    # ---------------------------------
    # Capture radius visual sphere (follows the evader)
    # ---------------------------------
    def _create_capture_visual(self):
        stage = sim_utils.SimulationContext.instance().stage
        path = "/World/Visuals/CaptureRadius"
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            return

        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr(float(self.cfg.capture_radius))

        xform = UsdGeom.Xformable(sphere.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))

        # Make it translucent-ish by using preview surface (simple)
        mat = self._make_preview_material(stage, "/World/Materials/CaptureBlue", (0.2, 0.4, 1.0))
        UsdShade.MaterialBindingAPI(sphere.GetPrim()).Bind(mat)

        # Make it not interfere with physics
        # (No collision API applied)

    def _update_capture_visual(self):
        stage = sim_utils.SimulationContext.instance().stage
        path = "/World/Visuals/CaptureRadius"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid():
            return

        ev = self._robots[self.EVADER].data.root_pos_w  # (N,3)
        p = ev[0].tolist()  # env_0

        xform = UsdGeom.Xformable(prim)
        ops = xform.GetOrderedXformOps()
        # Ensure we have a translate op
        if len(ops) == 0:
            xform.AddTranslateOp().Set(Gf.Vec3d(*p))
        else:
            ops[0].Set(Gf.Vec3d(*p))

    # ---------------------------------
    # Camera follow evader
    # ---------------------------------
    def _create_follow_camera(self):
        stage = sim_utils.SimulationContext.instance().stage
        cam_path = "/World/CameraFollow"
        prim = stage.GetPrimAtPath(cam_path)
        if prim.IsValid():
            return

        UsdGeom.Camera.Define(stage, cam_path)

    def _update_follow_camera(self):
        if not self.cfg.enable_camera_follow:
            return

        stage = sim_utils.SimulationContext.instance().stage
        cam_path = "/World/CameraFollow"
        cam = stage.GetPrimAtPath(cam_path)
        if not cam.IsValid():
            return

        ev = self._robots[self.EVADER].data.root_pos_w
        ev_pos = ev[0]  # env_0

        off = torch.tensor(self.cfg.camera_offset, device=self.device, dtype=torch.float)
        cam_pos = ev_pos + off

        # Translate camera
        xform = UsdGeom.Xformable(cam)
        ops = xform.GetOrderedXformOps()
        if len(ops) == 0:
            t_op = xform.AddTranslateOp()
            r_op = xform.AddRotateXYZOp()
        else:
            t_op = ops[0]
            # second op might not exist
            r_op = ops[1] if len(ops) > 1 else xform.AddRotateXYZOp()

        t_op.Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))

        # Point camera toward evader with a simple yaw/pitch
        look = (ev_pos - cam_pos)
        yaw = math.degrees(math.atan2(float(look[1]), float(look[0])))
        dist_xy = math.sqrt(float(look[0]**2 + look[1]**2)) + 1e-6
        pitch = -math.degrees(math.atan2(float(look[2]), dist_xy))

        r_op.Set(Gf.Vec3f(float(pitch), 0.0, float(yaw)))

        # Force viewport to use it (UI only)
        try:
            omni.kit.commands.execute("SetViewportCamera", camera_path=cam_path)
        except Exception:
            pass

    
    # ---------------------------------
    # Color-code the drones (hi red, lo green, evader black)
    # ---------------------------------
    def _color_code_drones(self):
        stage = sim_utils.SimulationContext.instance().stage

        mat_red = self._make_preview_material(stage, "/World/Materials/PursuerHiRed", (1.0, 0.2, 0.2))
        mat_grn = self._make_preview_material(stage, "/World/Materials/PursuerLoGreen", (0.2, 1.0, 0.2))
        mat_blk = self._make_preview_material(stage, "/World/Materials/EvaderBlack", (0.05, 0.05, 0.05))

        # Bind to each robot prim (works best when prim paths are explicit)
        # If you use regex env paths, for num_envs=1 these will typically exist:
        self._bind_material_recursive(stage, "/World/envs/env_0/Robot_0", mat_red)
        self._bind_material_recursive(stage, "/World/envs/env_0/Robot_1", mat_grn)
        self._bind_material_recursive(stage, "/World/envs/env_0/Robot_2", mat_blk)
