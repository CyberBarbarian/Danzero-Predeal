# baselines 说明

该目录存放掼蛋项目当前的基线（baseline）实现，主要分为两类：

```
baselines/
  README.md               # 本文件
  rule/                   # 规则型（基于启发式/手写逻辑）的旧版客户端实现集合
    ai1/ ai2/ ai3/ ai4/ ai6/
  legacy/                 # 已归档、暂不维护但保留参考价值的实现
    mc/                   # Monte Carlo / 简易 Q 网络旧基线（接口不再暴露）
```

## 1. rule/ 规则基线
| 目录 | 角色 | 状态 | 备注 |
|------|------|------|------|
| ai1  | 复杂规则集 | 保留 | 作为主要规则基线（推荐） |
| ai2  | 分阶段策略 | 保留 | 作为第二参考规则基线 |
| ai3  | 对抗/试验版 | 待裁剪 | 逻辑冗余，主要用于早期对抗测试 |
| ai4  | 与 ai1 高度重合 | 待归档 | 计划后续精简或删除 |
| ai6  | 另一套启发式 | 待归档 | 仅保留以便特定对比 |

统一特征：
- 入口方法仍为 `received_message(message)`，沿用旧通信协议（非标准 Gym/RL 接口）。
- 大量重复的牌型拆分、评估、排序逻辑（后续会抽取进入公共特征模块）。
- 已添加 `TODO` 注释标记需要迁移/抽象的位置。

## 2. legacy/
目前仅包含：
- `mc/`：旧 Monte Carlo / 简易强化学习基线，代码结构与新 Ray 训练流水线不兼容，已从 `agent_cls` 注册表移除。

保留原因：
- 供参考早期网络结构 / 数据流（可能用于快速对比）。
- 方便回溯旧实验。

未来处理：如 2 个版本后未再引用将整体删除或打包归档。

## 3. 命名与计划
后续将逐步把 `aiX` 形式重命名为统一前缀：
- `rule_ai1`, `rule_ai2` … 以减少语义不清。
- 或抽象成单一 `rule_agent` + 可配置策略 profile（若差异只在参数）。

## 4. 接口统一规划 (BaseAgent)
计划引入统一接口（草案）：
```python
class BaseAgent:
    def reset(self, env_info: dict):
        ...
    def act(self, obs, legal_action_mask=None) -> int:
        ...
    def observe(self, transition):  # 可选（训练用）
        ...
```
适配步骤：
1. 在 `rule/adapter.py` 增加包装器，将旧 `received_message` → `act` 适配。
2. 将牌型枚举/拆分函数抽取到 `guandan/env/utils.py` 或新建 `guandan/feature/combination.py`。
3. 行为空间（action space）固定化：构建统一动作索引表 + 合法掩码生成。
4. 逐步删除各 `aiX` 中重复 util。

## 5. 当前注册表 (agent_cls)
`guandan/agent/agents.py` 中保留：`ai1, ai2, ai3, ai4, ai6, torch, random`。
- `mc` 已移除（legacy）。
- 新训练/评估管线接入前，仅临时继续使用旧键名。

## 6. 推荐使用
临时直接引用：
```python
from guandan.agent.agents import agent_cls
agent = agent_cls['ai1'](id=0)
```
后续版本中将鼓励：
```python
from guandan.agent import make_agent  # 计划新增工厂函数
```

## 7. TODO 汇总
- [ ] 建立 `BaseAgent` 抽象与适配层
- [ ] 抽取公共牌型/特征逻辑
- [ ] 统一动作索引 + 合法掩码
- [ ] 重命名 aiX → rule_aiX 或合并策略配置
- [ ] 清理冗余 (ai3 / ai4 / ai6) 与日志噪声输出
- [ ] 为规则基线增加最小 smoke test（import + 简单假消息驱动）

## 8. 删除策略建议
| 条件 | 动作 |
|------|------|
| BaseAgent 适配完成且 rule_ai1 & rule_ai2 稳定 | 移除 ai3/ai4/ai6 |
| 无人再引用 legacy/mc 且新 RL 成功跑通 | 删除 legacy/mc |

---
如需在 README 中补充更多内部指标或重构进度，请追加 Issue 或在 PR 中更新此文件。

