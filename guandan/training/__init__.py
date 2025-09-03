"""Training framework package for DanZero (Guandan).

Modules:
  config            - use guandan.config (kept at top-level for backward compatibility)
  logger            - lightweight experiment logger
  checkpoint        - save/load model & optimizer states
  parameter_server  - parameter storage & broadcast (Ray stub)
  rollout_worker    - environment interaction (Ray stub)
  learner           - gradient update loop (Ray stub)
  ray_app           - orchestration entrypoint to wire everything together

This is an initial skeleton; concrete algorithm logic will be implemented incrementally.
"""
from .. import config  # re-export parser for convenience

__all__ = ["config"]

