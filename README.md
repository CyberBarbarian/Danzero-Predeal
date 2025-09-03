# DanZero (掼蛋强化学习与规则基线重构)

> 本仓库基于 DouZero 代码骨架进行二次演化，目标从原始“斗地主”环境迁移到“掼蛋 (Guandan)”并构建可扩展的分布式强化学习训练与评估框架。

## 1. 项目愿景
- 提供标准化、可向量化的掼蛋环境（状态/动作编码统一、可并行 rollout）。
- 整合 Ray 分布式组件，实现高并发自博弈采样与集中式 Learner 更新。
- 兼容规则基线 (rule-based baselines) 与学习型策略 (RL policy) 的统一评测接口。
- 渐进式清理历史斗地主遗留与重复工具函数，保障代码整洁与可维护性。

## 2. 当前进展摘要
| 模块 | 状态 | 说明 |
|------|------|------|
| env/ | 已存在 | 掼蛋核心逻辑（发牌 / 出牌合法性 / 桌面状态）初步可用；后续需补齐统一 Observation 构造与合法动作掩码。 |
| agent/baselines/rule | 已迁移 | 旧规则 AI (ai1/ai2/ai3/ai4/ai6) 迁入 rule/，添加 TODO；等待抽象化。 |
| agent/baselines/legacy/mc | 已归档 | 旧 MC/Q 基线，仅存档不再注册。 |
| agent/torch | 保留 | 老版神经网络样例（特征编码可供参考，将择机重构/下线）。 |
| training/ (Ray) | 初版 | 含 parameter_server / rollout_worker / learner 基础骨架，需与新 env.obs 接口对齐。 |
| 根 README | 已更新 | 聚焦掼蛋方向；旧 DouZero README 归档至 archive/doudizhu。 |
| baselines/README.md | 已撰写 | 说明规则基线与 legacy 状态及重构计划。 |

## 3. 目录结构 (精简视图)
```
guandan/
  env/                # 掼蛋游戏状态、规则与动作生成逻辑
  agent/
    agents.py         # 暂存注册表（后续以工厂/entrypoint 重构）
    random_agent.py   # 随机基线
    baselines/
      rule/           # 旧 ai1…ai6 规则基线 (迁移 + TODO 标记)
      legacy/mc/      # 已归档 MC/Q 基线
    torch/            # 老策略/特征示例 (待评估是否保留)
  training/           # Ray 训练流水线组件
archive/
  doudizhu/           # 旧斗地主脚本与原 README 归档
README.md             # 本文件
```

## 4. 规则基线策略 (rule/) 状态
| 基线 | 定位 | 下一步 |
|------|------|--------|
| ai1  | 主力复杂规则 | 适配 BaseAgent、抽出牌型解析模块 |
| ai2  | 分阶段决策 | 与 ai1 对齐动作索引；合并重复特征 |
| ai3  | 试验/对抗 | 评估是否删减或转 profile 参数化 |
| ai4  | 与 ai1 重合 | 计划移除 / 合并 |
| ai6  | 另一套启发式 | 若无独特价值则淘汰 |
| random | 随机基线 | 保持最简 sanity check |

详见 baselines/README.md。

## 5. 即将实施的重构里程碑
| 里程碑 | 目标 | 关键产出 |
|--------|------|----------|
| M1: Observation 统一 | 规范 obs dict(tensor) + legal_action_mask | env/observation_builder.py (新) |
| M2: Action 编码表 | 固定全局动作枚举 + 动态合法掩码生成 | action_space.json / builder |
| M3: BaseAgent 抽象 | 统一 act()/reset() 接口 | agent/base.py + rule 适配器 |
| M4: Ray 集成打通 | rollout → buffer → learner 闭环 | 训练脚本 & metrics 输出 |
| M5: 冗余清理 | 删除 ai3/ai4/ai6 & torch(可选) | 精简 agents.py / 文档更新 |

## 6. 技术设计要点 (规划)
### 6.1 Observation 组成（拟定）
| 分量 | 描述 | 形状 (示例) |
|------|------|------------|
| hand_cards | 自己当前手牌 one-hot/多通道 | (C1,) 或 (Suits,Ranks) |
| public_info | 桌面最近 N 轮出牌结构化编码 | (N, action_feat) |
| teammate_hint | 队友剩余牌/类别估计 | (vector) |
| phase_meta | 当前级数 / 进还贡阶段标记 | (small vector) |
| legal_mask | 合法动作掩码 (延迟构建) | (A,) |

### 6.2 Action 编码策略
- 建立全局“归一化动作集合”：按牌型 + 主标值 + 附属牌槽位展开。
- 提供 encode(move)->index 与 decode(index)->move 双向映射。
- 对“动态长度”牌型（顺子 / 三连对）使用长度上限 + padding；或拆分成 (牌型, 起始rank, 长度) 三元组组合编码。

### 6.3 BaseAgent 适配器
旧规则基线保留原 `received_message` 流程 → 适配器解析消息 → 临时构造 obs + mask → 调用内部决策 → 输出统一动作 index。

## 7. 快速开始 (当前临时方式)
```python
from guandan.agent.agents import agent_cls
agent = agent_cls['ai1'](id=0)
# 假设已有服务器推送的 JSON 消息 msg
action_index = agent.received_message(msg)
```
后续将改为：
```python
from guandan.agent import make_agent
agent = make_agent('rule_ai1')
obs, mask = env.reset()
while not done:
    a = agent.act(obs, mask)
    obs, reward, done, info = env.step(a)
```

## 8. 贡献与协作
欢迎通过 Issue / PR：
- 报告掼蛋规则/判定差异
- 提供更高效的动作空间压缩方案
- 优化特征抽取与网络结构

## 9. 变更日志（简版）
| 日期 | 变更 | 摘要 |
|------|------|------|
| 2025-09-02 | README 重写 | 归档旧 DouZero README，新增掼蛋方向说明 |
| 2025-09-02 | baselines 重组 | 规则 ai1~ai6 迁移，mc 归档 |

> 完整历史参见 Git 提交记录。

## 10. 未来讨论话题 (Open Questions)
- 顺子 / 连对 等可变长度动作的最优编码策略（固定槽 vs. 分段动作）
- 团队信息共享建模（对家牌信息估计的特征融合）
- 对局阶段自适应价值基准 (stage-aware value normalization)
- 数据回放 (experience replay) 与 DMC 异步优势比较

---
若你正在寻找旧 DouZero (斗地主) 说明，请查看：`archive/doudizhu/README_doudizhu_original.md`。

