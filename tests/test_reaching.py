from __future__ import annotations

import importlib
import sys
import types

import torch

from franka_wrist_camera_scene.tasks.reaching import ReachingTaskSpec, make_reaching_episode_spec


def _install_isaaclab_shims() -> None:
    isaaclab_module = types.ModuleType("isaaclab")
    isaaclab_module.__path__ = []

    assets_module = types.ModuleType("isaaclab.assets")
    assets_module.Articulation = object

    scene_module = types.ModuleType("isaaclab.scene")
    scene_module.InteractiveScene = object

    utils_module = types.ModuleType("isaaclab.utils")
    utils_module.__path__ = []

    math_module = types.ModuleType("isaaclab.utils.math")

    def quat_apply(_quat_wxyz: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return vec

    math_module.quat_apply = quat_apply

    sys.modules["isaaclab"] = isaaclab_module
    sys.modules["isaaclab.assets"] = assets_module
    sys.modules["isaaclab.scene"] = scene_module
    sys.modules["isaaclab.utils"] = utils_module
    sys.modules["isaaclab.utils.math"] = math_module


class _FakeObjectData:
    def __init__(self, root_pos_w: torch.Tensor) -> None:
        self.root_pos_w = root_pos_w


class _FakeObject:
    def __init__(self, root_pos_w: torch.Tensor) -> None:
        self.data = _FakeObjectData(root_pos_w)


class _FakeRobotData:
    def __init__(self, body_pose_w: torch.Tensor) -> None:
        self.body_pose_w = body_pose_w


class _FakeRobot:
    def __init__(self, body_pose_w: torch.Tensor) -> None:
        self.data = _FakeRobotData(body_pose_w)
        self.device = torch.device("cpu")

    def find_bodies(self, _name: str) -> tuple[list[int], list[str]]:
        return [0], ["panda_hand"]


class _FakeScene:
    def __init__(self, robot: _FakeRobot, target: _FakeObject) -> None:
        self.num_envs = 1
        self.env_origins = torch.zeros((1, 3))
        self._items = {"robot": robot, "target_cube": target}

    def __getitem__(self, key: str):
        return self._items[key]


def test_reaching_task_spec_preserves_tcp_offset() -> None:
    base_spec = ReachingTaskSpec(tcp_offset_local=(0.0, 0.0, 0.12))

    episode_spec = make_reaching_episode_spec(
        base_spec=base_spec,
        object_xy_offset=(0.02, -0.03),
        object_label="apple",
    )

    assert episode_spec.tcp_offset_local == (0.0, 0.0, 0.12)
    assert episode_spec.object_pos_local == (0.6, -0.19, 1.08)
    assert episode_spec.instruction == "reach the apple"


def test_reaching_success_measures_tcp_position() -> None:
    _install_isaaclab_shims()
    success_module = importlib.import_module("franka_wrist_camera_scene.episode.success")
    importlib.reload(success_module)

    robot = _FakeRobot(
        body_pose_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
    )
    target = _FakeObject(root_pos_w=torch.tensor([[0.0, 0.0, 0.10]], dtype=torch.float32))
    scene = _FakeScene(robot=robot, target=target)
    spec = ReachingTaskSpec(tcp_offset_local=(0.0, 0.0, 0.10))

    success = success_module.reaching_success(scene, spec)

    assert success.tolist() == [True]


def test_reaching_policy_uses_configured_tcp_offset(monkeypatch) -> None:
    _install_isaaclab_shims()
    policy_module = importlib.import_module("franka_wrist_camera_scene.policies.reaching_scripted")
    importlib.reload(policy_module)

    recorded: dict[str, torch.Tensor] = {}

    class FakeMotion:
        @classmethod
        def from_limits(
            cls,
            start_pos_w: torch.Tensor,
            goal_pos_w: torch.Tensor,
            quat_w: torch.Tensor,
            start_time_s: float,
            max_speed_m_s: float,
            max_accel_m_s2: float,
        ):
            recorded["goal_pos_w"] = goal_pos_w.clone()
            recorded["quat_w"] = quat_w.clone()
            return cls()

        def sample(self, sim_time_s: float):
            return (
                recorded["goal_pos_w"],
                recorded["quat_w"],
                True,
            )

    monkeypatch.setattr(policy_module, "LinearPoseMotion", FakeMotion)

    robot = _FakeRobot(
        body_pose_w=torch.tensor([[[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]], dtype=torch.float32),
    )
    target = _FakeObject(root_pos_w=torch.tensor([[0.50, -0.20, 1.00]], dtype=torch.float32))
    scene = _FakeScene(robot=robot, target=target)
    spec = ReachingTaskSpec(
        tcp_offset_local=(0.0, 0.0, 0.12),
        pregrasp_height_m=0.16,
    )

    policy = policy_module.ReachingScriptedPolicy(spec=spec)
    policy.bind(scene=scene, robot=robot)
    command = policy.step(obs=None, sim_time_s=0.0)

    expected_goal = torch.tensor([[0.50, -0.20, 1.04]], dtype=torch.float32)
    assert torch.allclose(recorded["goal_pos_w"], expected_goal)
    assert command.done is False
