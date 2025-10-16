#!/usr/bin/env python3
"""
Analyze Guandan training metrics with proper handling of competitive rewards.

Key insights:
- Rewards are zero-sum (sum to 0 across all 4 agents)
- Track per-agent and per-team separately
- Focus on win rates, rank distributions, and reward magnitudes
"""

import json
import glob
import os
from collections import defaultdict
from datetime import datetime

# Visualization
import matplotlib
matplotlib.use("Agg")  # Ensure headless rendering
import matplotlib.pyplot as plt

def extract_agent_rewards_from_learners(learners_dict):
    """Extract per-agent rewards from result['learners'][agent]['episode_return_mean']."""
    agent_rewards = {}
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        if isinstance(learners_dict, dict) and agent_id in learners_dict:
            agent_data = learners_dict.get(agent_id, {})
            if isinstance(agent_data, dict):
                val = agent_data.get('episode_return_mean')
                if isinstance(val, dict):
                    agent_rewards[agent_id] = val.get('value', val.get('mean', 0))
                elif val is not None:
                    agent_rewards[agent_id] = float(val)
    return agent_rewards

def calculate_team_metrics(agent_rewards):
    """
    Calculate team-based metrics.
    Team 1: agent_0 + agent_2
    Team 2: agent_1 + agent_3
    """
    team_metrics = {}
    
    if 'agent_0' in agent_rewards and 'agent_2' in agent_rewards:
        team_metrics['team_1_reward'] = agent_rewards['agent_0'] + agent_rewards['agent_2']
    
    if 'agent_1' in agent_rewards and 'agent_3' in agent_rewards:
        team_metrics['team_2_reward'] = agent_rewards['agent_1'] + agent_rewards['agent_3']
    
    # Calculate win probability based on which team has higher reward
    if 'team_1_reward' in team_metrics and 'team_2_reward' in team_metrics:
        team_metrics['team_1_winning'] = team_metrics['team_1_reward'] > team_metrics['team_2_reward']
        team_metrics['team_2_winning'] = team_metrics['team_2_reward'] > team_metrics['team_1_reward']
        team_metrics['reward_balance'] = abs(team_metrics['team_1_reward'] - team_metrics['team_2_reward'])
    
    return team_metrics

def analyze_ray_results(result_file):
    """Analyze Ray's result.json with proper Guandan metrics."""
    
    results = []
    try:
        with open(result_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        return None
    
    if not results:
        return None
    
    # Analyze across all iterations
    analysis = {
        'total_iterations': len(results),
        'agent_rewards_history': defaultdict(list),
        'team_rewards_history': defaultdict(list),
        'loss_history': defaultdict(list),
        'q_value_history': defaultdict(list),
        'gradient_norm_history': defaultdict(list),
        'win_counts': {'team_1': 0, 'team_2': 0},
    }
    
    for result in results:
        iteration = result.get('training_iteration', 0)
        # RLlib new structure uses 'learners' instead of 'learner_results'
        learners = result.get('learners', {}) if isinstance(result, dict) else {}
        # Also read env_runners for per-agent returns when learners-based reward is missing
        env_runners = result.get('env_runners', {}) if isinstance(result, dict) else {}
        
        # Extract per-agent metrics (prefer env_runners episodic returns over learners stats)
        agent_rewards = {}
        if isinstance(env_runners, dict):
            aer = env_runners.get('agent_episode_returns_mean')
            if isinstance(aer, dict):
                for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
                    val = aer.get(agent_id)
                    if isinstance(val, dict):
                        val = val.get('value', val.get('mean', 0))
                    if val is not None:
                        agent_rewards[agent_id] = float(val)
        # Fallback to learners if env_runners not available
        if not agent_rewards:
            agent_rewards = extract_agent_rewards_from_learners(learners)
        for agent_id, reward in agent_rewards.items():
            analysis['agent_rewards_history'][agent_id].append({
                'iteration': iteration,
                'reward': reward
            })
        
        # Calculate team metrics
        team_metrics = calculate_team_metrics(agent_rewards)
        for metric_name, value in team_metrics.items():
            if metric_name in ['team_1_reward', 'team_2_reward', 'reward_balance']:
                analysis['team_rewards_history'][metric_name].append({
                    'iteration': iteration,
                    'value': value
                })
        
        if 'team_1_winning' in team_metrics:
            if team_metrics['team_1_winning']:
                analysis['win_counts']['team_1'] += 1
            elif team_metrics['team_2_winning']:
                analysis['win_counts']['team_2'] += 1
        
        # Extract loss, Q-values, and gradient norms per agent from 'learners'
        for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
            if isinstance(learners, dict) and agent_id in learners:
                agent_data = learners[agent_id]
                if isinstance(agent_data, dict):
                    # Loss
                    if 'loss' in agent_data:
                        loss = agent_data['loss']
                        if isinstance(loss, dict):
                            loss = loss.get('value', loss.get('mean', 0))
                        analysis['loss_history'][agent_id].append({
                            'iteration': iteration,
                            'loss': float(loss) if loss is not None else 0
                        })
                    
                    # Q-values
                    if 'q_values_mean' in agent_data:
                        q_val = agent_data['q_values_mean']
                        if isinstance(q_val, dict):
                            q_val = q_val.get('value', q_val.get('mean', 0))
                        analysis['q_value_history'][agent_id].append({
                            'iteration': iteration,
                            'q_value': float(q_val) if q_val is not None else 0
                        })
                    
                    # Gradient norms
                    if 'gradients_adam_global_norm' in agent_data:
                        grad_norm = agent_data['gradients_adam_global_norm']
                        if isinstance(grad_norm, dict):
                            grad_norm = grad_norm.get('value', grad_norm.get('mean', 0))
                        analysis['gradient_norm_history'][agent_id].append({
                            'iteration': iteration,
                            'grad_norm': float(grad_norm) if grad_norm is not None else 0
                        })
    
    return analysis

def print_analysis_summary(analysis):
    """Print a human-readable summary of the analysis."""
    
    print("=" * 80)
    print("🎮 Guandan Training Analysis")
    print("=" * 80)
    print()
    
    print(f"📊 Total Iterations: {analysis['total_iterations']}")
    print()
    
    # Agent rewards summary
    print("💰 Agent Reward Statistics:")
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        if agent_id in analysis['agent_rewards_history']:
            rewards = [r['reward'] for r in analysis['agent_rewards_history'][agent_id]]
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                max_reward = max(rewards)
                min_reward = min(rewards)
                latest_reward = rewards[-1] if rewards else 0
                
                print(f"  {agent_id}:")
                print(f"    Latest: {latest_reward:+.4f} | Avg: {avg_reward:+.4f} | Range: [{min_reward:+.4f}, {max_reward:+.4f}]")
    print()
    
    # Team statistics
    print("🏆 Team Performance:")
    total_games = sum(analysis['win_counts'].values())
    if total_games > 0:
        team1_winrate = analysis['win_counts']['team_1'] / total_games * 100
        team2_winrate = analysis['win_counts']['team_2'] / total_games * 100
        
        print(f"  Team 1 (agent_0 + agent_2): {analysis['win_counts']['team_1']:4d} wins ({team1_winrate:.1f}%)")
        print(f"  Team 2 (agent_1 + agent_3): {analysis['win_counts']['team_2']:4d} wins ({team2_winrate:.1f}%)")
        print(f"  Balance: {'✅ Good' if 40 <= team1_winrate <= 60 else '⚠️  Imbalanced'}")
    print()
    
    # Loss trends
    print("📉 Loss Trends (Latest → First):")
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        if agent_id in analysis['loss_history']:
            losses = [l['loss'] for l in analysis['loss_history'][agent_id]]
            if len(losses) >= 2:
                latest = losses[-1]
                first = losses[0]
                improvement = ((first - latest) / first * 100) if first > 0 else 0
                
                print(f"  {agent_id}: {latest:.6f} (started: {first:.6f}, improvement: {improvement:+.1f}%)")
    print()
    
    # Q-value stability
    print("🎯 Q-Value Statistics:")
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        if agent_id in analysis['q_value_history']:
            q_vals = [q['q_value'] for q in analysis['q_value_history'][agent_id]]
            if q_vals:
                latest_q = q_vals[-1]
                avg_q = sum(q_vals) / len(q_vals)
                stability = "✅ Stable" if abs(latest_q) < 10 else "⚠️  Check values"
                
                print(f"  {agent_id}: Latest: {latest_q:+.4f} | Avg: {avg_q:+.4f} | {stability}")
    print()
    
    # Gradient health
    print("⚡ Gradient Norms (Latest):")
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        if agent_id in analysis['gradient_norm_history']:
            grad_norms = [g['grad_norm'] for g in analysis['gradient_norm_history'][agent_id]]
            if grad_norms:
                latest_norm = grad_norms[-1]
                avg_norm = sum(grad_norms) / len(grad_norms)
                
                if 0.01 <= latest_norm <= 1.0:
                    status = "✅ Healthy"
                elif latest_norm > 10:
                    status = "⚠️  Too high (unstable)"
                else:
                    status = "⚠️  Too low (slow learning)"
                
                print(f"  {agent_id}: {latest_norm:.6f} (avg: {avg_norm:.6f}) | {status}")
    print()
    
    print("=" * 80)

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _plot_agent_series(analysis, key: str, ylabel: str, title: str, out_path: str) -> None:
    plt.figure(figsize=(10, 6))
    plotted = 0
    for agent_id in ['agent_0', 'agent_1', 'agent_2', 'agent_3']:
        series = analysis.get(key, {}).get(agent_id, [])
        if not series:
            continue
        xs = [p['iteration'] for p in series]
        ys_key = 'loss' if key == 'loss_history' else 'q_value' if key == 'q_value_history' else 'grad_norm'
        if key == 'agent_rewards_history':
            ys = [p['reward'] for p in series]
        else:
            ys = [p[ys_key] for p in series]
        plt.plot(xs, ys, label=agent_id)
        plotted += 1
    if plotted == 0:
        plt.close()
        if os.path.exists(out_path):
            try:
                os.remove(out_path)
            except OSError:
                pass
        return
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def _plot_team_series(analysis, plot_dir: str) -> None:
    team_hist = analysis.get('team_rewards_history', {})
    # Plot team rewards
    plt.figure(figsize=(10, 6))
    plotted = 0
    for name in ['team_1_reward', 'team_2_reward']:
        series = team_hist.get(name, [])
        if not series:
            continue
        xs = [p['iteration'] for p in series]
        ys = [p['value'] for p in series]
        plt.plot(xs, ys, label=name)
        plotted += 1
    if plotted == 0:
        plt.close()
        team_rewards_path = os.path.join(plot_dir, 'team_rewards.png')
        if os.path.exists(team_rewards_path):
            try:
                os.remove(team_rewards_path)
            except OSError:
                pass
    else:
        plt.title('Team Rewards over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'team_rewards.png'))
        plt.close()

    # Plot reward balance (absolute difference)
    series = team_hist.get('reward_balance', [])
    if series:
        plt.figure(figsize=(10, 4))
        xs = [p['iteration'] for p in series]
        ys = [p['value'] for p in series]
        plt.plot(xs, ys, color='tab:purple')
        plt.title('Reward Differential |team_1 - team_2| over Iterations')
        plt.xlabel('Iteration')
        plt.ylabel('|Δ reward|')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'reward_differential.png'))
        plt.close()

    # Compute cumulative team-1 win rate if possible from team rewards history
    t1 = {p['iteration']: p['value'] for p in team_hist.get('team_1_reward', [])}
    t2 = {p['iteration']: p['value'] for p in team_hist.get('team_2_reward', [])}
    common_iters = sorted(set(t1.keys()) & set(t2.keys()))
    rates_x: list = []
    rates_y: list = []
    if common_iters:
        wins = 0
        for idx, it in enumerate(common_iters, start=1):
            if t1[it] > t2[it]:
                wins += 1
            rates_x.append(it)
            rates_y.append(wins / idx)
    else:
        # Fallback: infer winners from team_1_reward sign if team_2 data missing/misaligned
        if t1:
            iters_sorted = sorted(t1.keys())
            wins = 0
            for idx, it in enumerate(iters_sorted, start=1):
                if t1[it] > 0:
                    wins += 1
                rates_x.append(it)
                rates_y.append(wins / idx)

    if rates_x:
        plt.figure(figsize=(10, 4))
        plt.plot(rates_x, [r * 100 for r in rates_y], color='tab:green', marker='o', linewidth=2)
        plt.title('Cumulative Team-1 Win Rate')
        plt.xlabel('Iteration')
        plt.ylabel('Win rate (%)')
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'team1_winrate_cumulative.png'))
        plt.close()
    else:
        winrate_path = os.path.join(plot_dir, 'team1_winrate_cumulative.png')
        if os.path.exists(winrate_path):
            try:
                os.remove(winrate_path)
            except OSError:
                pass

def create_plots(analysis, plot_dir: str) -> None:
    _ensure_dir(plot_dir)
    # Per-agent rewards
    _plot_agent_series(analysis, 'agent_rewards_history', 'Reward', 'Agent Rewards over Iterations', os.path.join(plot_dir, 'agent_rewards.png'))
    # Per-agent loss
    _plot_agent_series(analysis, 'loss_history', 'Loss', 'Per-Agent Loss over Iterations', os.path.join(plot_dir, 'agent_loss.png'))
    # Per-agent Q-values
    _plot_agent_series(analysis, 'q_value_history', 'Q value', 'Per-Agent Q-Values over Iterations', os.path.join(plot_dir, 'agent_q_values.png'))
    # Per-agent gradient norms
    _plot_agent_series(analysis, 'gradient_norm_history', 'Grad norm', 'Per-Agent Gradient Norms over Iterations', os.path.join(plot_dir, 'agent_grad_norms.png'))
    # Team-level plots
    _plot_team_series(analysis, plot_dir)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Guandan training metrics")
    parser.add_argument('--ray-dir', type=str, help='Path to Ray results directory')
    parser.add_argument('--output', type=str, help='Save analysis to JSON file')
    parser.add_argument('--plot-dir', type=str, help='Directory to save plots (default: <ray_dir>/plots)')
    
    args = parser.parse_args()
    
    # Find Ray results
    if args.ray_dir:
        ray_dir = args.ray_dir
    else:
        # Auto-detect latest
        # Prefer project-local results/ray_results; fallback to ~/ray_results
        proj_results_ray = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "ray_results"))
        ray_results_base = proj_results_ray if os.path.isdir(proj_results_ray) else os.path.expanduser("~/ray_results")
        pattern = os.path.join(ray_results_base, "DMC_guandan_ma_*")
        dirs = glob.glob(pattern)
        
        if not dirs:
            print("❌ No Ray training directories found")
            return
        
        ray_dir = max(dirs, key=os.path.getmtime)
        print(f"🔍 Auto-detected: {ray_dir}\n")
    
    result_file = os.path.join(ray_dir, "result.json")
    
    if not os.path.exists(result_file):
        print(f"❌ No result.json found in {ray_dir}")
        return
    
    # Analyze
    print("📊 Analyzing training results...\n")
    analysis = analyze_ray_results(result_file)
    
    if analysis is None:
        print("❌ No valid results found")
        return
    
    # Print summary
    print_analysis_summary(analysis)
    
    # Save to file if requested
    if args.output:
        # Convert defaultdict to regular dict for JSON serialization
        output_data = {
            'total_iterations': analysis['total_iterations'],
            'agent_rewards_history': dict(analysis['agent_rewards_history']),
            'team_rewards_history': dict(analysis['team_rewards_history']),
            'loss_history': dict(analysis['loss_history']),
            'q_value_history': dict(analysis['q_value_history']),
            'gradient_norm_history': dict(analysis['gradient_norm_history']),
            'win_counts': analysis['win_counts'],
            'analysis_time': datetime.now().isoformat()
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n💾 Analysis saved to: {args.output}")

    # Create visualizations
    if args.plot_dir:
        plot_dir = args.plot_dir
    else:
        # If an output file is specified, prefer saving plots next to it under a 'plots' subdir
        if args.output:
            base_dir = os.path.dirname(os.path.abspath(args.output))
            plot_dir = os.path.join(base_dir, 'plots')
        else:
            plot_dir = os.path.join(ray_dir, 'plots')
    create_plots(analysis, plot_dir)
    print(f"🖼️  Plots saved to: {plot_dir}")

if __name__ == "__main__":
    main()


