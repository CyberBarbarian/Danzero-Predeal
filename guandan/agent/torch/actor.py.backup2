import numpy as np
import pickle
import torch
import io
from agent.torch.model import MLPActorCritic, MLPQNetwork


ActionNumber = 2


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


class Player():
    def __init__(self) -> None:
        # 模型初始化
        # self.model_id = args.iter * 5000
        self.model = MLPActorCritic((ActionNumber, 516+ActionNumber * 54), ActionNumber)
        with open('/aiarena/nas/guandan_douzero/guandan/agent/torch/ppo20000.pth', 'rb') as f:
            new_weights = CPU_Unpickler(f).load()
        # print('load model:', self.model_id)
        self.model.set_weights(new_weights)
        self.model_q = MLPQNetwork(567)
        with open('/aiarena/nas/guandan_douzero/guandan/agent/torch/q_network.ckpt', 'rb') as f:
            tf_weights = pickle.load(f)
        self.model_q.load_tf_weights(tf_weights)

    def sample(self, state) -> int:
        states = state['x_batch']
        legal_index = np.ones(ActionNumber)
        state_no_action = state['x_no_action']
        if len(states) >= ActionNumber:
            indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions))
        elif len(states) < ActionNumber:
            legal_action = len(states)
            legal_index[legal_action:] = np.zeros(ActionNumber-legal_action)
            top_indexs = self.model_q.get_max_n_index(states, ActionNumber)
            dqn_states = np.asarray(states[top_indexs])
            top_actions = dqn_states[:, -54:].flatten()
            states = np.concatenate((state_no_action, top_actions)) # 把动作先添加进来
            supple = -1 * np.ones(54 * (ActionNumber - legal_action))
            states = np.concatenate((states, supple))
            indexs = list(range(ActionNumber))

        action = self.model.step(states, legal_index)
        return indexs[action]
