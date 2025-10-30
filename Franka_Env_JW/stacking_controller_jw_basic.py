# isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller_jw.py

from typing import Optional, List
import math
import numpy as np

import isaacsim.robot.manipulators.controllers as manipulators_controllers
from isaacsim.core.prims import SingleArticulation
from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

# <<< Wähle dein Quaternion-Format der OBS hier aus: "xyzw" (Core/Stage) oder "wxyz" (deine Utils)
OBS_QUAT_FMT = "xyzw"   # ändere auf "wxyz", falls deine observations so liefern


class StackingController_JW(manipulators_controllers.StackingController):
    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        picking_order_cube_names: List[str],
        robot_observation_name: str,
    ) -> None:
        super().__init__(
            name=name,
            pick_place_controller=PickPlaceController(
                name=name + "_pick_place_controller",
                gripper=gripper,
                robot_articulation=robot_articulation,
            ),
            picking_order_cube_names=picking_order_cube_names,
            robot_observation_name=robot_observation_name,
        )

    # ---------- Helpers ----------
    @staticmethod
    def _to_xyzw(q: np.ndarray, fmt: str) -> np.ndarray:
        q = np.asarray(q, dtype=float).ravel()
        if fmt == "xyzw":
            return q
        elif fmt == "wxyz":
            # [w,x,y,z] -> [x,y,z,w]
            return np.array([q[1], q[2], q[3], q[0]], dtype=float)
        else:
            raise ValueError(f"Unsupported quat fmt: {fmt}")

    @staticmethod
    def _yaw_from_xyzw(q_xyzw: np.ndarray) -> float:
        x, y, z, w = float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        # Normierung auf [-pi, pi]
        return (yaw + math.pi) % (2.0 * math.pi) - math.pi

    def forward(
        self,
        observations: dict,
        end_effector_orientation: Optional[np.ndarray] = None,
        end_effector_offset: Optional[np.ndarray] = None,
    ) -> ArticulationAction:
        """
        Nutzt die Base-Class voll aus. Wenn keine EE-Orientierung übergeben wurde,
        wird sie aus der Würfel-Yaw gebaut:  ee = yaw ∘ down  (beides XYZW),
        wobei 'down' = (roll=0, pitch=π, yaw=0) den Greifer nach unten richtet.
        """

        # Wenn bereits eine Orientierung vorgegeben ist → direkt an Basisklasse weiterreichen
        if end_effector_orientation is not None:
            return super().forward(
                observations=observations,
                end_effector_orientation=end_effector_orientation,
                end_effector_offset=end_effector_offset,
            )

        # ---------- lokale Helfer (self-contained) ----------
        def to_xyzw_from_wxyz(q_wxyz: np.ndarray) -> np.ndarray:
            q = np.asarray(q_wxyz, dtype=float).ravel()
            w, x, y, z = q
            return np.array([x, y, z, w], dtype=float)

        def yaw_from_xyzw(q_xyzw: np.ndarray) -> float:
            x, y, z, w = map(float, q_xyzw)
            siny_cosp = 2.0 * (w * z + x * y)
            cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
            import math as _m
            return (_m.atan2(siny_cosp, cosy_cosp) + _m.pi) % (2.0 * _m.pi) - _m.pi

        def quat_mul_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
            # a,b: [x,y,z,w] → returns [x,y,z,w]
            ax, ay, az, aw = map(float, a)
            bx, by, bz, bw = map(float, b)
            return np.array(
                [
                    aw * bx + ax * bw + ay * bz - az * by,
                    aw * by - ax * bz + ay * bw + az * bx,
                    aw * bz + ax * by - ay * bx + az * bw,
                    aw * bw - ax * bx - ay * by - az * bz,
                ],
                dtype=float,
            )

        # ---------- aktuelle Cube-Orientierungen holen ----------
        # aktueller Würfelname (aus Base-StackingController)
        cube_name = self._picking_order_cube_names[self._current_cube]
        cob = observations[cube_name]

        # Deine Logs zeigen wxyz → zuerst in xyzw konvertieren
        pick_q_wxyz = cob.get("orientation")              # erwartet [w,x,y,z]
        place_q_wxyz = cob.get("target_orientation")      # optional [w,x,y,z]

        pick_q_xyzw = to_xyzw_from_wxyz(pick_q_wxyz) if pick_q_wxyz is not None else None
        place_q_xyzw = to_xyzw_from_wxyz(place_q_wxyz) if place_q_wxyz is not None else None

        # Fallbacks & Yaw extrahieren
        import math
        pick_yaw = yaw_from_xyzw(pick_q_xyzw) if pick_q_xyzw is not None else 0.0
        place_yaw = yaw_from_xyzw(place_q_xyzw) if place_q_xyzw is not None else pick_yaw

        # Phase bestimmen (aus Base-PickPlaceController)
        event = self._pick_place_controller.get_current_event()
        yaw = pick_yaw if event < 5 else place_yaw

        # ---------- EE-Quaternion robust zusammensetzen: yaw ∘ down ----------
        down_q = euler_angles_to_quat(np.array([0.0, math.pi, 0.0], dtype=float))        # Pitch = π
        yaw_q = euler_angles_to_quat(np.array([0.0, 0.0, float(yaw)], dtype=float))      # Z-Rotation
        ee_quat = quat_mul_xyzw(yaw_q, down_q)  # erst yaw, dann „Greifer nach unten“

        # (optional) Debug-Log gedrosselt
        try:
            self._dbg = getattr(self, "_dbg", 0) + 1
            if self._dbg % 20 == 0:
                import logging
                logging.getLogger("FrankaCubeStacking").info(
                    f"[{self.name}] evt={event} pick_yaw={pick_yaw:.3f} place_yaw={place_yaw:.3f} → yaw={yaw:.3f}, "
                    f"ee_quat(xyzw)={np.round(ee_quat,3)}"
                )
        except Exception:
            pass

        # An Basisklasse weiterreichen (die ruft intern den Pick&Place-Controller)
        return super().forward(
            observations=observations,
            end_effector_orientation=ee_quat,
            end_effector_offset=end_effector_offset,
        )
