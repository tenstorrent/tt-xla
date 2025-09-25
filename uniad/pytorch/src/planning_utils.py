# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
import copy
from typing import Any, Dict, Sequence, Tuple
import numpy as np
import numpy.typing as npt
from casadi import DM, Opti, OptiSol, sumsqr, vertcat, exp

Pose = Tuple[float, float, float]


class CollisionNonlinearOptimizer:
    """
    Optimize planned trajectory with predicted occupancy
    Solved with direct multiple-shooting.
    modified from https://github.com/motional/nuplan-devkit
    :param trajectory_len: trajectory length
    :param dt: timestep (sec)
    """

    def __init__(
        self, trajectory_len: int, dt: float, sigma, alpha_collision, obj_pixel_pos
    ):
        """
        :param trajectory_len: the length of trajectory to be optimized.
        :param dt: the time interval between trajectory points.
        """
        self.dt = dt
        self.trajectory_len = trajectory_len
        self.current_index = 0
        self.sigma = sigma
        self.alpha_collision = alpha_collision
        self.obj_pixel_pos = obj_pixel_pos
        self._dts: npt.NDArray[np.float32] = np.asarray([[dt] * trajectory_len])
        self._init_optimization()

    def _init_optimization(self) -> None:
        """
        Initialize related variables and constraints for optimization.
        """
        self.nx = 2

        self._optimizer = Opti()
        self._create_decision_variables()
        self._create_parameters()
        self._set_objective()

        self._optimizer.solver(
            "ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"}
        )

    def set_reference_trajectory(self, reference_trajectory: Sequence[Pose]) -> None:
        """
        Set the reference trajectory that the smoother is trying to loosely track.
        :param x_curr: current state of size nx (x, y)
        :param reference_trajectory: N x 3 reference, where the second dim is for (x, y)
        """
        self._optimizer.set_value(self.ref_traj, DM(reference_trajectory).T)
        self._set_initial_guess(reference_trajectory)

    def set_solver_optimizerons(self, options: Dict[str, Any]) -> None:
        """
        Control solver options including verbosity.
        :param options: Dictionary containing optimization criterias
        """
        self._optimizer.solver("ipopt", options)

    def solve(self) -> OptiSol:
        """
        Solve the optimization problem. Assumes the reference trajectory was already set.
        :return Casadi optimization class
        """
        return self._optimizer.solve()

    def _create_decision_variables(self) -> None:
        """
        Define the decision variables for the trajectory optimization.
        """
        self.state = self._optimizer.variable(self.nx, self.trajectory_len)
        self.position_x = self.state[0, :]
        self.position_y = self.state[1, :]

    def _create_parameters(self) -> None:
        """
        Define the expert trjactory and current position for the trajectory optimizaiton.
        """
        self.ref_traj = self._optimizer.parameter(2, self.trajectory_len)

    def _set_objective(self) -> None:
        """Set the objective function. Use care when modifying these weights."""
        alpha_xy = 1.0
        cost_stage = alpha_xy * sumsqr(
            self.ref_traj[:2, :] - vertcat(self.position_x, self.position_y)
        )

        alpha_collision = self.alpha_collision

        cost_collision = 0
        normalizer = 1 / (2.507 * self.sigma)
        for t in range(len(self.obj_pixel_pos)):
            x, y = self.position_x[t], self.position_y[t]
            for i in range(len(self.obj_pixel_pos[t])):
                col_x, col_y = self.obj_pixel_pos[t][i]
                cost_collision += (
                    alpha_collision
                    * normalizer
                    * exp(-((x - col_x) ** 2 + (y - col_y) ** 2) / 2 / self.sigma**2)
                )
        self._optimizer.minimize(cost_stage + cost_collision)

    def _set_initial_guess(self, reference_trajectory: Sequence[Pose]) -> None:
        """Set a warm-start for the solver based on the reference trajectory."""
        self._optimizer.set_initial(self.state[:2, :], DM(reference_trajectory).T)
