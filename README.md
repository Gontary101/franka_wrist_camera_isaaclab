# Franka wrist-camera tabletop scene for Isaac Lab

Clean Isaac Lab scene for a Franka Panda arm on a tabletop inside a warehouse background, with:

- a wrist-mounted RGB-D camera attached under `Robot/panda_hand/wrist_rgbd_camera`
- a fixed third-person “agent view” RGB-D camera
- a Seattle lab table, simple tabletop props, dome lighting, and a warehouse USD background
- a differential-IK controller that moves the gripper through a 40 cm horizontal circle above the table
- viewport markers showing the desired circular path and current IK target
- an optional wrist-camera pixel/depth probe for checking `(u, v, z)` image coordinates

The repo targets Isaac Sim 5.1 / Isaac Lab with Python 3.11 and your existing setup:

```bash
~/IsaacLab
conda env: env_isaaclab
```

## Run

```bash
unzip franka_wrist_camera_isaaclab.zip
cd franka_wrist_camera_isaaclab
conda activate env_isaaclab
./scripts/run.sh
```

Headless smoke run:

```bash
conda activate env_isaaclab
./scripts/run.sh --headless --max_steps 600
```

Custom Isaac Lab path:

```bash
ISAACLAB_ROOT=~/IsaacLab ./scripts/run.sh
```

## Circle IK test

The default gripper path is a 40 cm diameter circle in the air above the table:

```bash
./scripts/run.sh --circle_diameter 0.40 --circle_frequency 0.045
```

The path center, table height, robot base pose, and default camera geometry are centralized in:

```text
src/franka_wrist_camera_scene/settings.py
```

The IK control node is isolated in:

```text
src/franka_wrist_camera_scene/control.py
```

## Camera attachment note

The wrist-camera line is in `src/franka_wrist_camera_scene/scene.py`:

```python
prim_path="{ENV_REGEX_NS}/Robot/panda_hand/wrist_rgbd_camera"
```

That means the camera prim is created as a child of the Franka hand link, so it follows the wrist through the USD/physics hierarchy. The local camera pose is then set with `CameraCfg.OffsetCfg`, relative to `panda_hand`.

## Wrist camera coordinate probe

To visually verify the image coordinate convention:

```bash
./scripts/run.sh --probe_u 320 --probe_v 240 --save_probe_every 60
```

Images are saved under:

```text
camera_probes/
```

The convention is:

```python
z = depth[v, u]
```

where `u` is the image column, `v` is the image row, and `z` is `distance_to_image_plane` in meters.

## Files

```text
franka_wrist_camera_isaaclab/
├── README.md
├── pyproject.toml
├── scripts/
│   ├── run.sh
│   └── run_scene.py
└── src/
    └── franka_wrist_camera_scene/
        ├── __init__.py
        ├── camera_probe.py
        ├── control.py
        ├── scene.py
        ├── settings.py
        └── visualization.py
```

## Notes

- The scene uses Isaac Lab’s built-in Franka Panda high-PD config because it is intended for differential IK task-space control.
- The IK target uses the `panda_hand` body and the `panda_joint.*` joints.
- The robot starts from a stable tabletop-ready Franka pose and the controller immediately tracks a downward-facing gripper pose above the table.
- First launch can take time if Isaac Sim has to download or cache warehouse/table assets.
