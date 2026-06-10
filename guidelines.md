# Repository Coding Guidelines & Conventions

This document records the architectural standards and implementation guidelines established for the Franka Tabletop Isaac Lab project. Refer to this to prevent design drift, circular dependencies, or simulation setup corruption.

---

## 1. Decoupled Architecture

Keep the policy, trajectory generation, and controller loops strictly decoupled:

*   **Policies**: Policies (e.g., `CircleMotionPolicy`, `PickPlaceScriptedPolicy`) are finite-state machines or neural network steps. They must output a unified command structure using the `PolicyCommand` dataclass.
*   **Dataclasses / Commands**: `PolicyCommand` resides in `policies/scripted_base.py` and encapsulates:
    *   `target_pos_w`: Tensor representing target TCP position in world coordinates.
    *   `target_quat_w`: Tensor representing target TCP orientation in world coordinates.
    *   `finger_opening_m`: Total opening width of one finger (parallel gripper fingers target the same distance).
    *   `done`: Boolean flag indicating execution completion.
*   **IK Controller**: `CartesianIKController` in `control/ik.py` should remain general. It must **never** contain code relating to circles, specific task trajectories, or gripper commands. It simply consumes `target_pos_w` and `target_quat_w`, computes differential IK, and sets joint targets.
*   **Gripper Controller**: `GripperController` in `control/gripper.py` is dedicated to parallel finger controls.

---

## 2. Config Files & Settings

To prevent drift risk between scene layouts and task planners, establish a single source of truth:

*   **YAML Configuration**: Always mirror layout parameters (table heights, sizes, camera specifications, initial joint states) into `configs/scene.yaml`.
*   **No Redundant Settings**: [settings.py](src/franka_wrist_camera_scene/settings.py) dynamically reads constants using `load_yaml_config("scene.yaml")` from `utils/paths.py` to maintain compatibility without risking settings drift.
*   **Casing Conventions**: Use lowercase strings for conventions (e.g., `ros`, `world`) in configuration files to prevent parser mismatches inside Isaac Lab's camera and frame utilities.

---

## 3. Explicit Imports

*   **Keep Package Roots Empty**: To prevent submodules from becoming dependency magnets, keep the package `__init__.py` clean. 
*   **Explicit Submodule Imports**: Scripts and modules should import directly from the explicit submodule path (e.g., `from franka_wrist_camera_scene.control.ik import CartesianIKController`) rather than from the package root `__all__`.

---

## 4. Script Modularity

Main entry scripts (e.g., [debug_scene.py](scripts/debug_scene.py)) must remain lightweight and restricted to CLI parsing, pipeline setup, and the simulation step loop:

*   **Reset Logic**: episodic reset operations must be housed under `episode/reset.py` (e.g., `reset_robot_to_default(scene)`).
*   **Camera Warmup**: RTX-specific render prim offsets or warmup workarounds must be housed under `app/camera_warmup.py` (e.g., `nudge_camera_prims(sim, scene)`).

---

## 5. Isaac Lab Simulation Conventions

*   **Dynamic Rigid Bodies**: When creating movable objects (such as target manipulation cubes), spawn them using `RigidObjectCfg` instead of `AssetBaseCfg`.
*   **Geometry Configuration**: Specify physics properties directly in the shape configuration using `rigid_props=sim_utils.RigidBodyPropertiesCfg()` and `collision_props=sim_utils.CollisionPropertiesCfg()` (Note: the keyword argument is `rigid_props`, **not** `rigid_body_props`).
*   **TCP Alignment**: When target coordinates (like object pick poses) are defined in world coordinates, adjust wrist/hand commands by subtracting the TCP offset vector (`tcp_offset_w = quat_apply(quat_w, tcp_offset_local)`) to ensure the gripper matches the target's center instead of floating or penetrating the mesh.
*   **Wrist Camera Updates**: Keep the hand-mounted camera's `update_period` at `0.0` to force updates on every physics simulation step, eliminating camera coordinate lag relative to rapid link movements.
