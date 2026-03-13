# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

# Import your existing env/cfg so we reuse the robot + scene setup
from .peg_isaac_env import QuadcopterEnv, QuadcopterEnvCfg
from isaaclab.scene import InteractiveSceneCfg


class QuadcopterMotorEnvCfg(QuadcopterEnvCfg):
    """Same env, but action is 4 motor thrust commands."""

    # Action is still 4-dim, but now it means: [m1, m2, m3, m4]
    # where each mi is normalized in [0, 1] (we'll clamp it anyway).
    action_space = 4

    # Rough Crazyflie geometry (meters). Tune if needed.
    arm_length = 0.046  # ~46 mm from center to rotor (approx)
    # Yaw torque coefficient (N*m per N of thrust). This is a simplification.
    yaw_torque_coeff = 0.002

    # If True: treat incoming actions as [0,1]. If False: treat as [-1,1] and remap to [0,1].
    actions_are_01 = True


class QuadcopterMotorEnv(QuadcopterEnv):
    cfg: QuadcopterMotorEnvCfg

    def _pre_physics_step(self, actions: torch.Tensor):
        """
        actions: (num_envs, 4) motor commands.
        We'll convert motor thrusts -> net thrust + moments, then apply at body.
        """
        # Ensure shape
        if actions.shape[-1] != 4:
            raise ValueError(f"Expected actions with last dim 4 (4 motors), got {actions.shape}")

        # Convert to [0,1]
        if self.cfg.actions_are_01:
            u = actions.clone()
        else:
            u = (actions.clone() + 1.0) * 0.5

        u = u.clamp(0.0, 1.0)

        # Map normalized motor commands -> thrusts (Newtons)
        # Your original env defines max total thrust as:
        #   thrust_to_weight * weight
        # (see your existing mapping) :contentReference[oaicite:1]{index=1}
        max_total_thrust = self.cfg.thrust_to_weight * self._robot_weight
        max_per_motor = max_total_thrust / 4.0

        # Motor thrusts f1..f4 (N)
        f = u * max_per_motor  # (Nenv,4)

        # Net thrust (body +Z). We'll apply as external force on the body.
        total_thrust = torch.sum(f, dim=1, keepdim=True)  # (Nenv,1)
        self._thrust.zero_()
        self._thrust[:, 0, 2] = total_thrust.squeeze(-1)

        # Moments from a simple "+" quad model.
        # ASSUMED motor order:
        #   f1: front
        #   f2: right
        #   f3: back
        #   f4: left
        #
        # If your numbering is different, just permute f columns until roll/pitch match intuition.
        L = self.cfg.arm_length
        tau_x = L * (f[:, 3] - f[:, 1])          # roll: left - right
        tau_y = L * (f[:, 2] - f[:, 0])          # pitch: back - front

        # Yaw: depends on spin direction; simplified alternating signs.
        # Typical pattern: (+ - + -) for (f1,f2,f3,f4)
        k_yaw = self.cfg.yaw_torque_coeff
        tau_z = k_yaw * (f[:, 0] - f[:, 1] + f[:, 2] - f[:, 3])

        self._moment.zero_()
        self._moment[:, 0, 0] = tau_x
        self._moment[:, 0, 1] = tau_y
        self._moment[:, 0, 2] = tau_z
