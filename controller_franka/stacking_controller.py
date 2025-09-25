# stacking_controller.py
# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
import math
import numpy as np

from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.controllers.pick_place_controller import PickPlaceController


class StackingController(BaseController):
    """
    Stapel-Controller, der pro Würfel Pick-&-Place mit richtiger Yaw ausführt.

    Erwartete observation-Felder pro Würfel (alles optional, sinnvolle Defaults):
      - 'position': np.array([x,y,z])
      - 'target_position': np.array([X,Y,Z])
      - 'yaw' ODER 'orientation' (xyzw)
      - 'target_yaw' ODER 'target_orientation' (xyzw)
    """

    def __init__(
        self,
        name: str,
        pick_place_controller: PickPlaceController,
        picking_order_cube_names: typing.List[str],
        robot_observation_name: str,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._pick_place_controller = pick_place_controller
        self._picking_order_cube_names = picking_order_cube_names
        self._current_cube = 0
        self._robot_observation_name = robot_observation_name
        self.reset()

    def forward(
        self,
        observations: dict,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        end_effector_offset: typing.Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        if self._current_cube >= len(self._picking_order_cube_names):
            target_joint_positions = [None] * observations[self._robot_observation_name]["joint_positions"].shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        cube_name = self._picking_order_cube_names[self._current_cube]
        cube_obs = observations[cube_name]
        robot_obs = observations[self._robot_observation_name]

        # Yaw ermitteln (Pick/Place)
        pick_yaw = cube_obs.get("yaw", None)
        place_yaw = cube_obs.get("target_yaw", None)

        if pick_yaw is None:
            q_pick = cube_obs.get("orientation", None)  # erwartet [x,y,z,w]
            if q_pick is not None:
                pick_yaw = self._yaw_from_quat_xyzw(q_pick)
        if place_yaw is None:
            q_place = cube_obs.get("target_orientation", None)  # erwartet [x,y,z,w]
            if q_place is not None:
                place_yaw = self._yaw_from_quat_xyzw(q_place)

        if pick_yaw is None:
            pick_yaw = 0.0
        if place_yaw is None:
            place_yaw = pick_yaw

        # Optional: 90°-Raster aktivieren (auskommentiert)
        # snap = math.pi / 2.0
        # pick_yaw  = round(pick_yaw / snap)  * snap
        # place_yaw = round(place_yaw / snap) * snap

        actions = self._pick_place_controller.forward(
            picking_position=cube_obs["position"],
            placing_position=cube_obs["target_position"],
            current_joint_positions=robot_obs["joint_positions"],
            end_effector_orientation=end_effector_orientation,  # wenn gesetzt, überschreibt yaw-Logik
            end_effector_offset=end_effector_offset,
            picking_yaw=pick_yaw,
            placing_yaw=place_yaw,
        )

        if self._pick_place_controller.is_done():
            self._current_cube += 1
            self._pick_place_controller.reset()

        return actions

    def reset(self, picking_order_cube_names: typing.Optional[typing.List[str]] = None) -> None:
        self._current_cube = 0
        self._pick_place_controller.reset()
        if picking_order_cube_names is not None:
            self._picking_order_cube_names = picking_order_cube_names

    def is_done(self) -> bool:
        return self._current_cube >= len(self._picking_order_cube_names)

    # ----------------------------
    # Hilfsfunktionen
    # ----------------------------
    def _yaw_from_quat_xyzw(self, q: typing.Union[np.ndarray, typing.List[float]]) -> float:
        """
        Extrahiert Yaw (Rotation um Z) aus Quaternion im xyzw-Format.
        Falls deine Quelle wxyz liefert, vertausche die Indizes entsprechend.
        """
        x, y, z, w = float(q[0]), float(q[1]), float(q[2]), float(q[3])
        # Z-Yaw aus Quaternion (Tait-Bryan ZYX)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # Normierung auf [-pi, pi]
        return (yaw + math.pi) % (2.0 * math.pi) - math.pi
