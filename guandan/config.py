import argparse

# 简化后的 DanZero 配置参数（从原 dmc/arguments.py 抽取并裁剪）
parser = argparse.ArgumentParser(description='DanZero: Guandan AI Training Config')

# 通用实验设置
parser.add_argument('--xpid', default='danzero', help='实验ID')
parser.add_argument('--savedir', default='danzero_checkpoints', help='模型与日志根目录')
parser.add_argument('--save_interval', default=30, type=int, help='保存间隔(分钟)')
parser.add_argument('--seed', default=42, type=int, help='随机种子')

# 设备
parser.add_argument('--gpu_devices', default='0', type=str, help='使用的GPU编号, 逗号分隔')

# 训练主超参（与具体分布式实现无关，后续可被 Ray Worker 使用）
parser.add_argument('--total_frames', default=10000000, type=int, help='总训练帧数/步数目标')
parser.add_argument('--batch_size', default=32, type=int, help='Learner batch size')
parser.add_argument('--unroll_length', default=64, type=int, help='轨迹截断长度(T)')
parser.add_argument('--exp_epsilon', default=0.01, type=float, help='epsilon-greedy 探索概率')

# 优化相关
parser.add_argument('--learning_rate', default=1e-4, type=float, help='学习率')
parser.add_argument('--max_grad_norm', default=40.0, type=float, help='梯度裁剪阈值')

# 预留可扩展占位（后续需要再追加）
# parser.add_argument('--gamma', default=0.99, type=float, help='折扣因子(如需)')
# parser.add_argument('--entropy_coef', default=0.0, type=float, help='策略熵系数(如需)')

__all__ = ['parser']

