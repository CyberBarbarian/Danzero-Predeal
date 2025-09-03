# 项目结构 (DanZero 掼蛋版本)

本仓库现聚焦掼蛋(gamecore+多策略 Agent)。旧 DouZero 斗地主 DMC 训练框架已废弃并即将删除；新的训练骨架位于 `guandan/training/`。

## 顶层
- `train.py` 入口 -> 使用 `guandan.config` + `guandan.training.ray_app.train` (占位)。
- `requirements*.txt` 依赖（尚未加入 ray, 需要时自行添加）。
- `PROJECT_STRUCTURE.md` 本文件。
- 旧评估/脚本 (`evaluate.py`, `ADP_test.py`, etc.) 仍为斗地主遗留，后续可清理或迁移。

## guandan/
```
guandan/
  config.py              # 精简训练参数 (DanZero)
  training/              # 新训练骨架 (取代 dmc)
    logger.py            # 轻量 CSV 日志
    checkpoint.py        # 保存/加载权重 (占位)
    parameter_server.py  # 参数服务器占位
    rollout_worker.py    # 采样工作者占位
    learner.py           # 学习器占位
    ray_app.py           # 训练 orchestrator 占位
  env/                   # 掼蛋游戏核心逻辑
    game.py              # 环境 + 消息协议 + 自对弈驱动
    engine.py            # 规则/阶段流程 (发牌, 贡/还贡, 升级, 轮转)
    utils.py             # 合法动作生成 / 牌型分析 / 贡牌逻辑
    card_deck.py         # 两副牌生成与发牌
    player.py / context.py / table.py  # 状态数据结构
  agent/
    agents.py            # 代理注册表
    random_agent.py      # 随机策略
    ai*/                 # 多套规则/启发式策略实现
    torch/               # 深度模型相关 (旧 PPO / Q 网络)
    mc/                  # Monte Carlo 简易代理
  dmc/ (DEPRECATED)      # 已标记废弃; 保留空壳防止意外 import
```

## 废弃模块说明
- `guandan/dmc/`：原斗地主三角色分布式自博弈代码；与掼蛋四人队伍机制不兼容。`__init__.py` 已抛出 ImportError；确认无外部依赖后可物理删除。
- 依赖其中的牌编码函数已不再使用；需要的应重写为掼蛋专用实现。

## 下一步建议
1. 物理删除 `guandan/dmc/`（确认无引用后）。
2. 在 `training/` 内实现最小可运行训练循环：
   - RolloutWorker 封装 `Env.one_episode()` 收集 (obs, action, reward)。
   - Learner 定义模型/优化器并更新。
   - ParameterServer + 简单广播 (或直接单进程先行)。
3. 增加 `encoding/` 或 `features/` 模块，集中管理牌/动作特征，避免重复散落在 `utils.py` 中。
4. 引入测试 (pytest)：
   - 合法动作生成边界用例 (起手/进贡/接风/万能牌补缺)。
   - 升级与贡牌流程回归。
5. 清理未用斗地主脚本 & 依赖 (如 GitPython 若不再写 git 元数据)。

## 简要依赖图 (当前)
```
agent.* <--JSON--> env/game.Env --> env/engine.GameEnv --> env/utils + (card_deck/player/context/table)
train.py -> training.ray_app.train (占位)
```

## 命名统一
- 包名逐步从历史 `douzero` 语义迁移到 `danzero` (config 默认 xpid 已修改)。
- 推荐后续发布前更新 `setup.py` 与 README.

---
此文档已反映 dmc 废弃与新 training 架构，请在实现实际训练逻辑后再次更新。
