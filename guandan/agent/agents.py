# TODO: Unify BaseAgent interface later; rule-based agents are migration placeholders
from .baselines.rule.ai1.client import Ai1_agent
from .random_agent import RandomAgent
from .baselines.rule.ai2.client import Ai2_agent
from .baselines.rule.ai4.client import Ai4_agent
from .baselines.rule.ai3.client import Ai3_agent
from .baselines.rule.ai6.client import Ai6_agent
try:
    from .torch.client import Torch_agent
except Exception:
    Torch_agent = None


agent_cls = {
    'ai1': Ai1_agent,  # TODO: rename to rule_ai1
    'ai2': Ai2_agent,  # TODO: rename to rule_ai2
    'ai3': Ai3_agent,  # legacy rule variant
    'ai4': Ai4_agent,  # legacy rule variant
    'ai6': Ai6_agent,  # legacy rule variant
    'torch': Torch_agent,  # legacy NN baseline (may be None)
    'random': RandomAgent
}