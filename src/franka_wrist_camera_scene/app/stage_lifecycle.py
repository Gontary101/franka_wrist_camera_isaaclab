"""Isaac Sim stage lifecycle helpers for collection scripts."""

from __future__ import annotations

from collections.abc import Sequence

import isaaclab.sim as sim_utils
import omni.usd

TABLETOP_SCENE_PRIM_PATHS = ("/World/envs", "/World/Warehouse", "/World/Light")
STAGE_FLUSH_UPDATES = 2


def _flush_app_updates(simulation_app, count: int = STAGE_FLUSH_UPDATES) -> None:
    for _ in range(count):
        simulation_app.update()


def stop_simulation_for_stage_edit(sim: sim_utils.SimulationContext, simulation_app) -> None:
    sim._disable_app_control_on_stop_handle = True
    try:
        sim.stop()
        _flush_app_updates(simulation_app)
    finally:
        sim._disable_app_control_on_stop_handle = False


def delete_scene_prims(
    sim: sim_utils.SimulationContext,
    simulation_app,
    prim_paths: Sequence[str] = TABLETOP_SCENE_PRIM_PATHS,
) -> None:
    stop_simulation_for_stage_edit(sim, simulation_app)
    stage = omni.usd.get_context().get_stage()
    existing_paths = [path for path in prim_paths if stage.GetPrimAtPath(path)]
    if existing_paths:
        sim_utils.delete_prim(existing_paths, stage=stage)
        _flush_app_updates(simulation_app)
