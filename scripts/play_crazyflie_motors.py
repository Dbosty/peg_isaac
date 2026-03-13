
import argparse
from isaaclab.app import AppLauncher

# CLI
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch Isaac Sim / Omniverse
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch

from peg_isaac.tasks.direct.peg_isaac.peg_isaac_env_traj import (
    QuadcopterMotorEnv,
    QuadcopterMotorEnvCfg,
)

import numpy as np
from isaacsim.util.debug_draw import _debug_draw


class TrajectoryDrawer:
    """Draw desired + actual trajectories as persistent line segments in the viewport."""
    def __init__(self, thickness=2.0, draw_every=1):
        self.dd = _debug_draw.acquire_debug_draw_interface()
        self.thickness = np.array([float(thickness)], dtype=np.float32)
        self.draw_every = max(1, int(draw_every))
        self.prev_actual = None
        self.prev_desired = None
        self.k = 0

    def reset(self, clear=True):
        if clear:
            self.dd.clear_lines()
            self.dd.clear_points()
        self.prev_actual = None
        self.prev_desired = None
        self.k = 0

    def update(self, actual_xyz, desired_xyz):
        self.k += 1
        if self.k % self.draw_every != 0:
            return

        a = np.asarray(actual_xyz, dtype=np.float32).reshape(3)
        d = np.asarray(desired_xyz, dtype=np.float32).reshape(3)

        if self.prev_actual is not None:
            self.dd.draw_lines(
                np.array([self.prev_actual], dtype=np.float32),
                np.array([a], dtype=np.float32),
                np.array([[1.0, 0.2, 0.2]], dtype=np.float32),  # actual: red-ish
                self.thickness,
            )
        if self.prev_desired is not None:
            self.dd.draw_lines(
                np.array([self.prev_desired], dtype=np.float32),
                np.array([d], dtype=np.float32),
                np.array([[0.2, 0.8, 1.0]], dtype=np.float32),  # desired: blue-ish
                self.thickness,
            )

        self.prev_actual = a
        self.prev_desired = d


# ----------------- math helpers -----------------
def quat_to_rotmat(q_wxyz: torch.Tensor) -> torch.Tensor:
    # q: (N,4) in (w,x,y,z)
    w, x, y, z = q_wxyz[:, 0], q_wxyz[:, 1], q_wxyz[:, 2], q_wxyz[:, 3]
    R = torch.zeros((q_wxyz.shape[0], 3, 3), device=q_wxyz.device)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - z * w)
    R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w)
    R[:, 2, 1] = 2 * (y * z + x * w)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def yaw_from_rotmat(R_wb: torch.Tensor) -> torch.Tensor:
    # yaw from world-from-body rotation matrix
    # yaw = atan2(R21, R11) with standard ZYX convention
    return torch.atan2(R_wb[:, 1, 0], R_wb[:, 0, 0])

def wrap_pi(a: torch.Tensor) -> torch.Tensor:
    return (a + math.pi) % (2 * math.pi) - math.pi

def clamp(x, lo, hi):
    return torch.clamp(x, lo, hi)

def rot2d(theta: torch.Tensor) -> torch.Tensor:
    c = torch.cos(theta)
    s = torch.sin(theta)
    R = torch.zeros((theta.shape[0], 2, 2), device=theta.device)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    return R

# ----------------- trajectory -----------------
def circle_traj(t: float, R=0.75, w=0.35, z=1.0, center=(0.0, 0.0)):
    cx, cy = center
    px = cx + R * math.cos(w * t)
    py = cy + R * math.sin(w * t)
    pz = z
    vx = -R * w * math.sin(w * t)
    vy =  R * w * math.cos(w * t)
    vz = 0.0
    ax = -R * w * w * math.cos(w * t)
    ay = -R * w * w * math.sin(w * t)
    az = 0.0
    return (px, py, pz), (vx, vy, vz), (ax, ay, az)

# ----------------- cascaded PID (firmware-like) -----------------
class CascadePID:
    """
    Crazyflie-like cascade:
      Position PID -> velocity setpoint (in yaw-aligned XY, global Z)
      Velocity PID -> acceleration setpoint
      Accel/tilt mapping -> roll/pitch + thrust
      Attitude PID -> body rate setpoint
      Rate PID -> body moments
      Mixer -> 4 motor thrusts
    """

    def __init__(self, device, mass, g, arm_length, k_yaw):
        self.device = device
        self.m = mass
        self.g = g
        self.L = arm_length
        self.k = k_yaw

        # ----- Gains (good starting point for Crazyflie-size in sim) -----
        # Position -> velocity setpoint
        self.pos_kp_xy = 0.6
        self.pos_kp_z  = 1.5
        self.pos_ki_xy = 0.0
        self.pos_ki_z  = 0.0
        self.pos_kd_xy = 0.0
        self.pos_kd_z  = 0.0

        # Velocity -> acceleration setpoint
        self.vel_kp_xy = 1.0
        self.vel_kp_z  = 3.0
        self.vel_ki_xy = 0.0
        self.vel_ki_z  = 0.0
        self.vel_kd_xy = 0.0
        self.vel_kd_z  = 0.0

        # Attitude -> rate
        self.att_kp = torch.tensor([3.0, 3.0, 1.0], device=device)  # roll,pitch,yaw
        self.att_ki = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Rate -> moments
        self.rate_kp = torch.tensor([0.02, 0.02, 0.015], device=device)
        self.rate_ki = torch.tensor([0.0, 0.0, 0.0], device=device)

        # Limits (important for stability)
        self.vmax_xy = 0.8
        self.vmax_z  = 0.6
        self.amax_xy = 2.0
        self.amin_z  = -2.0
        self.amax_z  = 3.0
        self.tilt_max = math.radians(8.0)  # Crazyflie-ish

        # Integrators
        self.i_pos = torch.zeros((1, 3), device=device)
        self.i_vel = torch.zeros((1, 3), device=device)
        self.i_att = torch.zeros((1, 3), device=device)
        self.i_rate = torch.zeros((1, 3), device=device)

    def reset(self):
        self.i_pos.zero_()
        self.i_vel.zero_()
        self.i_att.zero_()
        self.i_rate.zero_()

    def step(self, dt, pos_w, vel_w, R_wb, ang_vel_b, p_des, v_des, a_ff, yaw_des):
        # Current yaw and yaw-aligned frame rotation (world -> yaw frame for XY)
        yaw = yaw_from_rotmat(R_wb)
        R_yaw = rot2d(-yaw)  # rotate world XY into yaw-aligned XY

        # Position error (world)
        e_p_w = p_des - pos_w
        e_v_w = v_des - vel_w

        # Convert XY errors to yaw-aligned frame (like CF position controller)
        e_p_xy = torch.bmm(R_yaw, e_p_w[:, 0:2].unsqueeze(-1)).squeeze(-1)
        e_v_xy = torch.bmm(R_yaw, e_v_w[:, 0:2].unsqueeze(-1)).squeeze(-1)

        # ---- Position PID -> velocity setpoint ----
        self.i_pos[:, 0:2] += e_p_xy * dt
        self.i_pos[:, 2:3] += e_p_w[:, 2:3] * dt

        v_sp_xy = self.pos_kp_xy * e_p_xy + self.pos_ki_xy * self.i_pos[:, 0:2] + self.pos_kd_xy * e_v_xy
        v_sp_z  = self.pos_kp_z  * e_p_w[:, 2:3] + self.pos_ki_z  * self.i_pos[:, 2:3] + self.pos_kd_z  * e_v_w[:, 2:3]

        v_sp_xy = clamp(v_sp_xy, -self.vmax_xy, self.vmax_xy)
        v_sp_z  = clamp(v_sp_z,  -self.vmax_z,  self.vmax_z)

        # Velocity setpoint is in yaw-frame XY; convert back to world XY
        R_yaw_inv = rot2d(yaw)
        v_sp_xy_w = torch.bmm(R_yaw_inv, v_sp_xy.unsqueeze(-1)).squeeze(-1)
        v_sp_w = torch.cat([v_sp_xy_w, v_sp_z], dim=1)

        # ---- Velocity PID -> acceleration setpoint ----
        e_v2 = v_sp_w - vel_w
        self.i_vel += e_v2 * dt

        a_sp = torch.zeros((1, 3), device=self.device)
        a_sp[:, 0:2] = self.vel_kp_xy * e_v2[:, 0:2] + self.vel_ki_xy * self.i_vel[:, 0:2] + self.vel_kd_xy * (0.0 * e_v2[:, 0:2])
        a_sp[:, 2:3] = self.vel_kp_z  * e_v2[:, 2:3] + self.vel_ki_z  * self.i_vel[:, 2:3] + self.vel_kd_z  * (0.0 * e_v2[:, 2:3])

        # Add feedforward accel
        a_sp += a_ff

        # Clamp accel
        a_sp[:, 0:2] = clamp(a_sp[:, 0:2], -self.amax_xy, self.amax_xy)
        a_sp[:, 2:3] = clamp(a_sp[:, 2:3], self.amin_z, self.amax_z)

        # ---- Accel -> thrust + desired roll/pitch (small-angle mapping with yaw) ----
        # Desired force in world
        F_des_w = self.m * (a_sp + torch.tensor([0.0, 0.0, self.g], device=self.device))
        # Thrust magnitude along body z
        z_b_w = R_wb[:, :, 2]
        T = torch.sum(F_des_w * z_b_w, dim=1, keepdim=True)
        T = clamp(T, 0.0, 2.5 * self.m * self.g)

        # Convert desired accel to desired tilt (firmware style: compute roll/pitch given yaw)
        # Using:
        # ax = g*(sin(pitch)*cos(yaw) + sin(roll)*sin(yaw))
        # ay = g*(sin(pitch)*sin(yaw) - sin(roll)*cos(yaw))
        # For small angles, solve approx:
        ax = a_sp[:, 0:1]
        ay = a_sp[:, 1:2]
        cy = torch.cos(yaw).unsqueeze(-1)
        sy = torch.sin(yaw).unsqueeze(-1)

        pitch_des = ( ax * cy + ay * sy ) / self.g
        roll_des  = ( ax * sy - ay * cy ) / self.g
        roll_des  = clamp(roll_des,  -self.tilt_max, self.tilt_max)
        pitch_des = clamp(pitch_des, -self.tilt_max, self.tilt_max)

        # ---- Attitude PID -> rate setpoint ----
        # Current roll/pitch/yaw from R_wb (ZYX)
        roll  = torch.atan2(R_wb[:, 2, 1], R_wb[:, 2, 2]).unsqueeze(-1)
        pitch = torch.asin(clamp(-R_wb[:, 2, 0], -1.0, 1.0)).unsqueeze(-1)
        yaw_e = wrap_pi((yaw_des - yaw).unsqueeze(-1))

        e_att = torch.cat([roll_des - roll, pitch_des - pitch, yaw_e], dim=1)
        self.i_att += e_att * dt

        rate_sp = self.att_kp * e_att + self.att_ki * self.i_att
        # (CF rates are fast; keep these sane)
        rate_sp = clamp(rate_sp, -torch.tensor([6.0, 6.0, 4.0], device=self.device),
                               torch.tensor([6.0, 6.0, 4.0], device=self.device))

        # ---- Rate PID -> moments ----
        e_rate = rate_sp - ang_vel_b
        self.i_rate += e_rate * dt

        tau = self.rate_kp * e_rate + self.rate_ki * self.i_rate
        # clamp moments (tune if saturating)
        tau = clamp(tau, -torch.tensor([0.03, 0.03, 0.02], device=self.device),
                         torch.tensor([0.03, 0.03, 0.02], device=self.device))

        tau_x, tau_y, tau_z = tau[:, 0:1], tau[:, 1:2], tau[:, 2:3]

        l = self.L / math.sqrt(2.0)
        k = self.k

        f1 = T/4.0 - tau_x/(4.0*l) + tau_y/(4.0*l) - tau_z/(4.0*k)   # FL
        f2 = T/4.0 - tau_x/(4.0*l) - tau_y/(4.0*l) + tau_z/(4.0*k)   # FR
        f3 = T/4.0 + tau_x/(4.0*l) - tau_y/(4.0*l) - tau_z/(4.0*k)   # RR
        f4 = T/4.0 + tau_x/(4.0*l) + tau_y/(4.0*l) + tau_z/(4.0*k)   # RL

        fN = torch.cat([f1, f2, f3, f4], dim=1)
        fN = clamp(fN, 0.0, 2.0 * self.m * self.g)
        return fN

        # ---- Mixer: [T, tau] -> motor thrusts (N), motor order: front, right, back, left ----
        # L = self.L
        # k = self.k

        # f1 = T/4.0 - tau_y/(2.0*L) + tau_z/(4.0*k)
        # f2 = T/4.0 - tau_x/(2.0*L) - tau_z/(4.0*k)
        # f3 = T/4.0 + tau_y/(2.0*L) + tau_z/(4.0*k)
        # f4 = T/4.0 + tau_x/(2.0*L) - tau_z/(4.0*k)

        # fN = torch.cat([f1, f2, f3, f4], dim=1)
        # fN = clamp(fN, 0.0, 2.0 * self.m * self.g)  # loose clamp
        # return fN

from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.sim as sim_utils


def define_markers() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/trajectoryMarkers",
        markers={
            "desired": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.0, 1.0, 0.0)
                ),
            ),
            "actual": sim_utils.SphereCfg(
                radius=0.04,
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(1.0, 0.0, 0.0)
                ),
            ),
        },
    )
    return VisualizationMarkers(marker_cfg)

import sys

def main():
    cfg = QuadcopterMotorEnvCfg()
    cfg.scene.num_envs = 1
    cfg.debug_vis = False
    cfg.actions_are_01 = True

    # IMPORTANT: give yourself control authority (prevents saturation/bounce)
    cfg.thrust_to_weight = 4.0

    env = QuadcopterMotorEnv(cfg, render_mode="human")
    obs, _ = env.reset()

    drawer = TrajectoryDrawer(thickness=2.0, draw_every=1)
    drawer.reset(clear=True)

    device = env.device
    dt = env.step_dt
    g = 9.81

    weight = float(env._robot_weight)
    m = weight / g

    controller = CascadePID(
        device=device,
        mass=m,
        g=g,
        arm_length=cfg.arm_length,
        k_yaw=cfg.yaw_torque_coeff,
    )
    controller.reset()

    max_total_thrust = cfg.thrust_to_weight * env._robot_weight
    max_per_motor = max_total_thrust / 4.0

    t = 0.0
    yaw_des = 0.0

    # Create markers
    my_visualizer = define_markers()

    # History for breadcrumb trails
    desired_history = []
    actual_history = []
    max_history = 500



    # while True:
    while simulation_app.is_running():
        # State
        pos_w = env._robot.data.root_pos_w
        quat = env._robot.data.root_quat_w
        R_wb = quat_to_rotmat(quat)

        # Your env gives BODY linear velocity; rotate to world
        vel_b = env._robot.data.root_lin_vel_b
        vel_w = torch.bmm(R_wb, vel_b.unsqueeze(-1)).squeeze(-1)
        ang_vel_b = env._robot.data.root_ang_vel_b

        # # Desired trajectory (circle)
        # (px, py, pz), (vx, vy, vz), (ax, ay, az) = circle_traj(t, R=0.75, w=0.35, z=1.0)
        # p_des = torch.tensor([[px, py, pz]], device=device)
        # v_des = torch.tensor([[vx, vy, vz]], device=device)
        # a_ff  = torch.tensor([[ax, ay, az]], device=device)

        # Hover in place
        p_des = torch.tensor([[0.0, 0.0, 1.0]], device=device)
        v_des = torch.zeros((1, 3), device=device)
        a_ff  = torch.zeros((1, 3), device=device)


        # Marker visualization
        desired_now = p_des[0].detach().cpu()
        actual_now = pos_w[0].detach().cpu()

        desired_history.append(desired_now.clone())
        actual_history.append(actual_now.clone())

        if len(desired_history) > max_history:
            desired_history.pop(0)
        if len(actual_history) > max_history:
            actual_history.pop(0)

        desired_pts = torch.stack(desired_history, dim=0)
        actual_pts = torch.stack(actual_history, dim=0)

        marker_locations = torch.cat([desired_pts, actual_pts], dim=0)

        marker_orientations = torch.zeros((marker_locations.shape[0], 4), dtype=torch.float32)
        marker_orientations[:, 0] = 1.0

        marker_indices = torch.cat([
            torch.zeros(desired_pts.shape[0], dtype=torch.int64),
            torch.ones(actual_pts.shape[0], dtype=torch.int64),
        ], dim=0)

        my_visualizer.visualize(
            marker_locations,
            marker_orientations,
            marker_indices=marker_indices,
        )

        # Controller -> motor thrusts in N
        fN = controller.step(dt, pos_w, vel_w, R_wb, ang_vel_b, p_des, v_des, a_ff, yaw_des)

        # Normalize to [0,1] motor commands for your motor env
        u = clamp(fN / max_per_motor, 0.0, 1.0)

        # Step (ignore resets while tuning trajectory control)
        obs, rew, terminated, truncated, info = env.step(u)
        t += dt

        # If you want auto-reset only on hard crash:
        # if pos_w[0,2].item() < 0.05:
        #     obs, _ = env.reset()
        #     controller.reset()
        #     t = 0.


if __name__ == "__main__":
    main()
    simulation_app.close()