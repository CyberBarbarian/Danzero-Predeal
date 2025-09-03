from typing import Any, Dict

class ParameterServer:
    """简单参数服务器占位实现.
    后续接入 Ray 时可: @ray.remote class ParameterServer: ...
    """
    def __init__(self):
        self._weights: Dict[str, Any] = {}
        self._step: int = 0

    def push(self, name: str, weights: Any, step: int):
        self._weights[name] = weights
        self._step = step

    def pull(self, name: str):
        return self._weights.get(name), self._step

__all__ = ['ParameterServer']

