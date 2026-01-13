import os
import logging

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})
os.environ.setdefault("PYTHONUNBUFFERED", "1")
log = logging.getLogger("Cloner_IsaacSim")

from isaacsim.core.cloner import Cloner    # import Cloner interface
from isaacsim.core.cloner import GridCloner    # import GridCloner interface
from isaacsim.core.utils.stage import get_current_stage
from pxr import UsdGeom
import numpy as np


# create our base environment with one cube
base_env_path = "/World/Cube_0"
UsdGeom.Cube.Define(get_current_stage(), base_env_path)
log.info(f"Created base environment with one cube at {base_env_path}")

# create a Cloner instance
cloner = GridCloner(spacing=3)
# cloner = Cloner()

cube_positions = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0], [9, 0, 0]])

# clone the cube at target paths at specified positions
# target_paths = cloner.generate_paths("/World/Cube", 4)
# cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths, positions=cube_positions)

# generate 4 paths that begin with "/World/Cube" - path will be appended with _{index}
target_paths = cloner.generate_paths("/World/Cube", 4)

# clone the cube at target paths
cloner.clone(source_prim_path="/World/Cube_0", prim_paths=target_paths)