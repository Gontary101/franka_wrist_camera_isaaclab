"""Isaac Sim and pxr compatibility patches and launcher setup."""

from __future__ import annotations

import sys
import types

# Compatibility layer for Isaac Sim 6.0 (redirects omni.physics.tensors.impl.api -> omni.physics.tensors.api)
class LazyApiModule(types.ModuleType):
    def __getattr__(self, name):
        import omni.physics.tensors.api as api
        return getattr(api, "DeformableBodyView" if name == "SoftBodyView" else name)

    def __dir__(self):
        import omni.physics.tensors.api as api
        return dir(api)

# Apply sys.modules patches immediately when this module is imported
if "omni.physics.tensors.impl.api" not in sys.modules:
    sys.modules["omni.physics.tensors.impl.api"] = LazyApiModule("omni.physics.tensors.impl.api")
if "omni.physics.tensors.impl" not in sys.modules:
    sys.modules["omni.physics.tensors.impl"] = types.ModuleType("omni.physics.tensors.impl")


def patch_physx_schema() -> None:
    """Apply pxr.PhysxSchema patch for compatibility after SimulationApp is started."""
    from pxr import PhysxSchema
    if not hasattr(PhysxSchema, "PhysxDeformableBodyAPI"):
        PhysxSchema.PhysxDeformableBodyAPI = PhysxSchema.PhysxRigidBodyAPI
