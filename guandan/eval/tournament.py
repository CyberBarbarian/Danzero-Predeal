import argparse
import os
from contextlib import redirect_stdout, redirect_stderr
from tqdm import tqdm

from guandan.env.context import Context
from guandan.env.game import Env
from guandan.agent.agents import agent_cls


def make_env_with_agents(agent_key: str) -> Env:
    ctx = Context()
    env = Env(ctx)
    # Override default agents: team (0,2) uses agent_key; team (1,3) uses random
    env.agent0 = agent_cls[agent_key](0)
    env.agent1 = agent_cls['random'](1)
    env.agent2 = agent_cls[agent_key](2)
    env.agent3 = agent_cls['random'](3)
    env.agents = [env.agent0, env.agent1, env.agent2, env.agent3]
    return env


def make_env_head_to_head(agent_a: str, agent_b: str) -> Env:
    ctx = Context()
    env = Env(ctx)
    # Team A on seats 0 & 2, Team B on seats 1 & 3
    env.agent0 = agent_cls[agent_a](0)
    env.agent1 = agent_cls[agent_b](1)
    env.agent2 = agent_cls[agent_a](2)
    env.agent3 = agent_cls[agent_b](3)
    env.agents = [env.agent0, env.agent1, env.agent2, env.agent3]
    return env


def evaluate_agent_vs_random(agent_key: str, num_games: int, quiet: bool = False) -> float:
    env = make_env_with_agents(agent_key)
    team02_wins = 0
    devnull = open(os.devnull, 'w')
    try:
        for _ in tqdm(range(num_games), desc=f"{agent_key} vs random", leave=False, disable=quiet):
            before = env.victory_num[0] + env.victory_num[2]
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env.one_episode()
            after = env.victory_num[0] + env.victory_num[2]
            if after > before:
                team02_wins += 1
            env.reset()
    finally:
        devnull.close()
    return team02_wins / float(num_games)


def evaluate_head_to_head(agent_a: str, agent_b: str, num_games: int, quiet: bool = False) -> float:
    env = make_env_head_to_head(agent_a, agent_b)
    team_a_wins = 0
    devnull = open(os.devnull, 'w')
    try:
        for _ in tqdm(range(num_games), desc=f"{agent_a} vs {agent_b}", leave=False, disable=quiet):
            before = env.victory_num[0] + env.victory_num[2]
            with redirect_stdout(devnull), redirect_stderr(devnull):
                env.one_episode()
            after = env.victory_num[0] + env.victory_num[2]
            if after > before:
                team_a_wins += 1
            env.reset()
    finally:
        devnull.close()
    return team_a_wins / float(num_games)


def main():
    parser = argparse.ArgumentParser(description="Evaluate ai1..ai6 vs random or head-to-head")
    parser.add_argument("--games", type=int, default=100, help="Number of games per agent")
    parser.add_argument("--agents", nargs="*", default=["ai1", "ai2", "ai3", "ai4", "ai6"], help="Agents to evaluate vs random")
    parser.add_argument("--opp", type=str, default=None, help="If set, run head-to-head: --agents <A> and --opp <B>")
    parser.add_argument("--quiet", action="store_true", help="Disable tqdm progress bars; only print final results")
    args = parser.parse_args()

    if args.opp is not None:
        if len(args.agents) != 1:
            raise SystemExit("For head-to-head, provide exactly one --agents entry (team A) and one --opp (team B)")
        a = args.agents[0]
        b = args.opp
        win_rate = evaluate_head_to_head(a, b, args.games, quiet=args.quiet)
        print(f"Head-to-head win rate (team 0&2={a} vs team 1&3={b}) over {args.games} games: {win_rate*100:.2f}%")
        return

    results = {}
    for agent_key in args.agents:
        win_rate = evaluate_agent_vs_random(agent_key, args.games, quiet=args.quiet)
        results[agent_key] = win_rate

    # Concise summary
    print("Agent vs Random win rates (team 0&2):")
    for k in args.agents:
        print(f"- {k}: {results[k]*100:.2f}% over {args.games} games")


if __name__ == "__main__":
    main()
