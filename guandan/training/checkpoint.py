import os, json, torch
from typing import Any, Dict

# 简单checkpoint工具，可根据需要扩展

def save_checkpoint(path: str, model_state: Dict[str, Any], optim_state: Dict[str, Any]=None, meta: Dict[str, Any]=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model_state': model_state,
        'optim_state': optim_state,
        'meta': meta
    }
    torch.save(payload, path)

    # 冗余写一个 json meta（非必须）
    if meta:
        meta_path = path + '.meta.json'
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def load_checkpoint(path: str):
    if not os.path.exists(path):
        return None
    return torch.load(path, map_location='cpu')

__all__ = ['save_checkpoint','load_checkpoint']

