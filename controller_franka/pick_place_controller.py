# pick_place_controller.py
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
import numpy as np

from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.stage import get_stage_units
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers.gripper import Gripper


class PickPlaceController(BaseController):
    """
    Ein einfacher Pick-&-Place-Zustandsautomat mit Yaw-Unterstützung.

    Phasen (je 1.0 s interne Zeit, dt je Phase in events_dt):
    - 0: EE über Ziel (Höhe h1)
    - 1: Absenken auf Greifhöhe h0
    - 2: Warten (Inertie)
    - 3: Greifer schließen
    - 4: Anheben auf h1
    - 5: Horizontal zum Ziel-XY interpolieren (+ Yaw-Interpolation)
    - 6: Vertikal auf Zielhöhe
    - 7: Greifer öffnen
    - 8: Anheben auf h1
    - 9: Zurück zum alten XY (Yaw bleibt)

    Args:
        name: Name
        cspace_controller: Controller, der EE-Ziele in Gelenkziele (ArticulationAction) mappt
        gripper: Greifer-Controller
        end_effector_initial_height: h1 (Default 0.3 m in Stage Units)
        events_dt: dt je Phase (max. 10 Einträge)
    """

    def __init__(
        self,
        name: str,
        cspace_controller: BaseController,
        gripper: Gripper,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        BaseController.__init__(self, name=name)
        self._event = 0
        self._t = 0.0
        self._h1 = end_effector_initial_height if end_effector_initial_height is not None else 0.3 / get_stage_units()
        self._h0 = None
        self._events_dt = events_dt
        if self._events_dt is None:
            self._events_dt = [0.008, 0.005, 0.1, 0.1, 0.0025, 0.001, 0.0025, 1.0, 0.008, 0.08]
        else:
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("events dt need to be list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")

        self._cspace_controller = cspace_controller
        self._gripper = gripper
        self._pause = False

        # Yaw-States
        self._picking_yaw = 0.0
        self._placing_yaw = 0.0

    def is_paused(self) -> bool:
        return self._pause

    def get_current_event(self) -> int:
        return self._event

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        picking_yaw: typing.Optional[float] = None,
        placing_yaw: typing.Optional[float] = None,
    ) -> ArticulationAction:
        """
        Führt einen Schritt aus.

        Args:
            picking_position: Objekt-Pose (Position) zum Greifen, im lokalen/Welt-Frame (konsistent zum C-Space-Controller)
            placing_position: Objekt-Pose (Position) zum Ablegen
            current_joint_positions: aktuelle Gelenkwinkel
            end_effector_offset: EE-Offset (xyz)
            end_effector_orientation: feste EE-Orientierung (überschreibt Yaw-Logik, wenn gesetzt)
            picking_yaw: gewünschte Yaw beim Greifen (um Z, rad)
            placing_yaw: gewünschte Yaw beim Ablegen (um Z, rad)
        """
        if end_effector_offset is None:
            end_effector_offset = np.array([0.0, 0.0, 0.0], dtype=float)

        if self._pause or self.is_done():
            self.pause()
            target_joint_positions = [None] * current_joint_positions.shape[0]
            return ArticulationAction(joint_positions=target_joint_positions)

        if self._event == 2:
            target_joint_positions = ArticulationAction(joint_positions=[None] * current_joint_positions.shape[0])
        elif self._event == 3:
            target_joint_positions = self._gripper.forward(action="close")
        elif self._event == 7:
            target_joint_positions = self._gripper.forward(action="open")
        else:
            # Ziele/Referenzen aktualisieren
            if self._event in [0, 1]:
                self._current_target_x = float(picking_position[0])
                self._current_target_y = float(picking_position[1])
                self._h0 = float(picking_position[2])

                # Yaw-Defaults setzen/aktualisieren
                if picking_yaw is not None:
                    self._picking_yaw = float(picking_yaw)
                if placing_yaw is not None:
                    self._placing_yaw = float(placing_yaw)
                else:
                    self._placing_yaw = self._picking_yaw

            interpolated_xy = self._get_interpolated_xy(
                placing_position[0], placing_position[1], self._current_target_x, self._current_target_y
            )
            target_height = self._get_target_hs(float(placing_position[2]))
            position_target = np.array(
                [
                    float(interpolated_xy[0]) + end_effector_offset[0],
                    float(interpolated_xy[1]) + end_effector_offset[1],
                    float(target_height) + end_effector_offset[2],
                ],
                dtype=float,
            )

            # Orientierung wählen:
            # - Wenn explizit gegeben: nutzen (kompatibel zu altem Verhalten).
            # - Sonst: Greifer nach unten (Pitch=π) + Yaw (um Z) je nach Phase.
            if end_effector_orientation is None:
                yaw_target = self._get_target_yaw()
                end_effector_orientation = euler_angles_to_quat(np.array([0.0, np.pi, yaw_target], dtype=float))

            target_joint_positions = self._cspace_controller.forward(
                target_end_effector_position=position_target, target_end_effector_orientation=end_effector_orientation
            )

        # interne Zeit/Phase fortschreiben
        self._t += self._events_dt[self._event]
        if self._t >= 1.0:
            self._event += 1
            self._t = 0.0

        return target_joint_positions

    # ----------------------------
    # Interne Hilfen
    # ----------------------------
    def _get_interpolated_xy(self, target_x, target_y, current_x, current_y):
        alpha = self._get_alpha()
        xy_target = (1.0 - alpha) * np.array([current_x, current_y]) + alpha * np.array([target_x, target_y])
        return xy_target

    def _get_alpha(self):
        if self._event < 5:
            return 0.0
        elif self._event == 5:
            return self._mix_sin(self._t)
        elif self._event in [6, 7, 8]:
            return 1.0
        elif self._event == 9:
            return 1.0
        else:
            raise ValueError()

    def _get_target_hs(self, target_height):
        if self._event == 0:
            h = self._h1
        elif self._event == 1:
            a = self._mix_sin(max(0.0, self._t))
            h = self._combine_convex(self._h1, self._h0, a)
        elif self._event == 3:
            h = self._h0
        elif self._event == 4:
            a = self._mix_sin(max(0.0, self._t))
            h = self._combine_convex(self._h0, self._h1, a)
        elif self._event == 5:
            h = self._h1
        elif self._event == 6:
            h = self._combine_convex(self._h1, target_height, self._mix_sin(self._t))
        elif self._event == 7:
            h = target_height
        elif self._event == 8:
            h = self._combine_convex(target_height, self._h1, self._mix_sin(self._t))
        elif self._event == 9:
            h = self._h1
        else:
            raise ValueError()
        return float(h)

    def _get_target_yaw(self) -> float:
        """Yaw je Phase: 0..4 Pick-Yaw, 5 smooth Interp., 6..9 Place-Yaw."""
        pick_yaw = float(getattr(self, "_picking_yaw", 0.0))
        place_yaw = float(getattr(self, "_placing_yaw", pick_yaw))
        if self._event < 5:
            return pick_yaw
        elif self._event == 5:
            a = self._mix_sin(self._t)
            return self._interp_angle(pick_yaw, place_yaw, a)
        else:
            return place_yaw

    def _interp_angle(self, a: float, b: float, alpha: float) -> float:
        d = self._wrap_to_pi(b - a)
        return self._wrap_to_pi(a + alpha * d)

    def _wrap_to_pi(self, ang: float) -> float:
        return (ang + np.pi) % (2.0 * np.pi) - np.pi

    def _mix_sin(self, t):
        return 0.5 * (1.0 - np.cos(t * np.pi))

    def _combine_convex(self, a, b, alpha):
        return (1.0 - alpha) * a + alpha * b

    def reset(
        self,
        end_effector_initial_height: typing.Optional[float] = None,
        events_dt: typing.Optional[typing.List[float]] = None,
    ) -> None:
        """Reset auf Phase 0."""
        BaseController.reset(self)
        self._cspace_controller.reset()
        self._event = 0
        self._t = 0.0
        if end_effector_initial_height is not None:
            self._h1 = end_effector_initial_height
        self._pause = False
        if events_dt is not None:
            self._events_dt = events_dt
            if not isinstance(self._events_dt, (np.ndarray, list)):
                raise Exception("event velocities need to be list or numpy array")
            if isinstance(self._events_dt, np.ndarray):
                self._events_dt = self._events_dt.tolist()
            if len(self._events_dt) > 10:
                raise Exception("events dt length must be less than 10")
        # Yaw zurücksetzen
        self._picking_yaw = 0.0
        self._placing_yaw = 0.0

    def is_done(self) -> bool:
        return self._event >= len(self._events_dt)

    def pause(self) -> None:
        self._pause = True

    def resume(self) -> None:
        self._pause = False
