from __future__ import annotations
import argparse
import subprocess
from pathlib import Path
import sys
import os


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=int, default=4, help="Anzahl paralleler Prozesse")
    p.add_argument("--base_logdir", type=Path, default=Path("./00_my_envs/Franka_Cube_Stacking_JW/logs"))
    p.add_argument("--start_seed", type=int, default=1000)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--cam_freq", type=int, default=20)
    p.add_argument("--cam_res", type=str, default="256x256")
    p.add_argument("--rotation_mode", type=str, default="yaw", choices=["yaw", "xyz"])
    p.add_argument("--yaw_min", type=float, default=-5.0)
    p.add_argument("--yaw_max", type=float, default=5.0)
    p.add_argument("--forward_axis", type=str, default="x", choices=["x", "y"])
    p.add_argument("--keep_cubes_rotated", action="store_true")
    p.add_argument("--rand_cube_rotation", action="store_true")
    p.add_argument("--material_seed", type=int, default=None)

    default_py = "./python.sh" if Path("./python.sh").exists() else sys.executable
    p.add_argument("--python", type=str, default=default_py, help="Python-Binary (z.B. ./python.sh)")
    here = Path(__file__).resolve().parent
    default_script = str((here / "stacking_runner.py").resolve())
    p.add_argument("--script", type=str, default=default_script, help="Runner-Skript-Pfad")
    return p.parse_args()


def main():
    args = parse_args()
    procs = []

    # Beispiel: ENV pro Prozess (hier leer, aber Stelle vorhanden)
    base_env = os.environ.copy()

    for i in range(args.runs):
        seed = args.start_seed + i
        logdir = args.base_logdir / f"run_{i:03d}"
        cmd = [
            args.python,
            args.script,
            "--headless" if args.headless else "",
            "--logdir", str(logdir),
            "--cam_freq", str(args.cam_freq),
            "--cam_res", args.cam_res,
            "--seed", str(seed),
            "--forward_axis", args.forward_axis,
            "--rotation_mode", args.rotation_mode,
            "--yaw_min", str(args.yaw_min),
            "--yaw_max", str(args.yaw_max),
        ]
        if args.keep_cubes_rotated:
            cmd.append("--keep_cubes_rotated")
        if args.rand_cube_rotation:
            cmd.append("--rand_cube_rotation")
        if args.material_seed is not None:
            cmd.extend(["--material_seed", str(args.material_seed + i)])

        cmd = [c for c in cmd if c != ""]
        print("[LAUNCH]", " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=base_env))

    for p in procs:
        p.wait()


if __name__ == "__main__":
    main()