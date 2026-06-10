"""Camera render-product warmup and RTX transform refresh helpers."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene


def nudge_camera_prims(sim: sim_utils.SimulationContext, scene: InteractiveScene) -> None:
    """Dirty camera transforms once to prevent white camera views."""
    from pxr import Gf, UsdGeom
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    for camera_name in ("wrist_camera", "agent_camera"):
        camera = scene[camera_name]
        for path in camera._view.prim_paths:
            prim = stage.GetPrimAtPath(path)
            xform = UsdGeom.Xformable(prim)
            translate_op = next(
                (op for op in xform.GetOrderedXformOps() if op.GetOpName() == "xformOp:translate"),
                None,
            )
            if translate_op is None:
                translate_op = xform.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble)

            original = translate_op.Get() or Gf.Vec3d(0.0, 0.0, 0.0)
            translate_op.Set(Gf.Vec3d(original[0] + 1.0e-3, original[1], original[2]))
            sim.render()
            translate_op.Set(original)

    sim.render()
