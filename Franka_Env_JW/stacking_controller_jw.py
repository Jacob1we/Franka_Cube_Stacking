# isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller_jw.py

from typing import Optional, List
import math
import logging
import numpy as np

# import isaacsim.robot.manipulators.controllers as manipulators_controllers
from .base_stacking_controller_jw import Base_StackingController
from isaacsim.core.prims import SingleArticulation
# from isaacsim.robot.manipulators.examples.franka.controllers.pick_place_controller import PickPlaceController
from .pick_place_controller_jw import PickPlaceController_JW as PickPlaceController

from isaacsim.robot.manipulators.grippers.parallel_gripper import ParallelGripper

# class StackingController_JW(manipulators_controllers.StackingController): #JW
class StackingController_JW(Base_StackingController): #JW
    """[summary]

    Args:
        name (str): [description]
        gripper (ParallelGripper): [description]
        robot_prim_path (str): [description]
        picking_order_cube_names (List[str]): [description]
        robot_observation_name (str): [description]
    """

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        picking_order_cube_names: List[str],
        robot_observation_name: str,
    ) -> None:
        self.log = logging.getLogger("StackingController")
        self.log.info(f"Initiation of Stacking Controller {name} is Done")
        # manipulators_controllers.StackingController.__init__( #JW
        Base_StackingController.__init__( #JW
            self,
            name=name,
            pick_place_controller=PickPlaceController(
                name=name + "_pick_place_controller", gripper=gripper, robot_articulation=robot_articulation
            ),
            picking_order_cube_names=picking_order_cube_names,
            robot_observation_name=robot_observation_name,
        )
        return
