from typing import Any, Dict, List

class RolloutWorker:
    """环境采样占位类.
    后续将包装掼蛋Env并执行step收集trajectory.
    """
    def __init__(self, wid: int):
        self.wid = wid

    def run_episode(self) -> Dict[str, Any]:
        # TODO: 接入真实环境, 返回: {'reward': float, 'steps': int, 'trajectories': List}
        return {'reward': 0.0, 'steps': 0, 'trajectories': []}

__all__ = ['RolloutWorker']

