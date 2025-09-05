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

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

from isaacsim.core.api import World
# from isaacsim.core.api.tasks import task
# exts/isaacsim.robot.manipulators.examples/franka/controllers/stacking_controller
from isaacsim.robot.manipulators.examples.franka.controllers.stacking_controller import StackingController
from isaacsim.robot.manipulators.examples.franka.tasks import Stacking

from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils
import numpy as np
import isaacsim.core.utils.prims as prim_utils


# class DinoWmSceneStacking():
my_world = World(stage_units_in_meters=1.0)
my_world.scene.add_default_ground_plane()
data_logger = my_world.get_data_logger()
my_task = Stacking()
my_world.add_task(my_task)
robot_name = my_task.get_params()["robot_name"]["value"]
my_franka = my_world.scene.get_object(robot_name)
my_world.reset()

def frame_logging_func(tasks, scene):
    return {
        "joint_positions": scene.my_franka.get_joint_positions().tolist(),
        "applied_joint_positions": scene.my_franka.get_applied_action().joint_positions.tolist(),
    }

camera = Camera(
    prim_path="/World/camera",
    position=np.array([0.48, -3.6, 1.8]),
    frequency=20,
    resolution=(256, 256),
    orientation=rot_utils.euler_angles_to_quats(np.array([0, 22.5, 90]), degrees=True),
)

distant_light = prim_utils.create_prim(
    "/World/Distant_Light",
    "DistantLight",
    position=np.array([1.0, 1.0, 1.0]),
    attributes={
        # "inputs:radius": 0.01,
        "inputs:intensity": 500,
        "inputs:color": (1.0, 1.0, 1.0)
    }
)

my_controller = StackingController(
    name="stacking_controller",
    gripper=my_franka.gripper,
    robot_articulation=my_franka,
    picking_order_cube_names=my_task.get_cube_names(),
    robot_observation_name=robot_name,
)

articulation_controller = my_franka.get_articulation_controller()

# def _on_logging_event(self, val):

#     world = self.get_world()
#     data_logger = world.get_data_logger() # a DataLogger object is defined in the World by default
#     if not world.get_data_logger().is_started():
#         robot_name = self._task_params["robot_name"]["value"]
#         target_name = self._task_params["target_name"]["value"]

#         # A data logging function is called at every time step index if the data logger is started already.
#         # We define the function here. The tasks and scene are passed to this function when called.

#         def frame_logging_func(tasks, scene):
#             return {
                
#                 "joint_positions": scene.get_object(robot_name).get_joint_positions().tolist(),# save data as lists since its a JSON file.
#                 "applied_joint_positions": scene.get_object(robot_name)
#                 .get_applied_action()
#                 .joint_positions.tolist(),
#                 "target_position": scene.get_object(target_name).get_world_pose()[0].tolist(),
#             }

#         data_logger.add_data_frame_logging_func(frame_logging_func) # adds the function to be called at each physics time step.
#     if val:
#         data_logger.start() # starts the data logging
#     else:
#         data_logger.pause()
#     return


i = 0
reset_needed = False
data_logger.add_data_frame_logging_func(frame_logging_func)
data_logger.start()

while simulation_app.is_running(): 
    my_world.step(render=True)

    if my_world.is_stopped() and not reset_needed: 
        reset_needed = True

    if my_world.is_playing(): 

        # wenn Stopp geklickt und noch nicht Stopp erfolgt
        if reset_needed:
            # Data Logger Output
            data_logger.save(log_path="./isaac_sim_data.json")
            data_logger.reset()
            data_logger.load(log_path="./isaac_sim_data.json")
            print(data_logger.get_data_frame(data_frame_index=2))
            
            my_world.reset()
            my_controller.reset()
            reset_needed = False

        observations = my_world.get_observations()
        actions = my_controller.forward(observations=observations)
        articulation_controller.apply_action(actions)

simulation_app.close()
