from __future__ import annotations
import itertools
import os
import time
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from pathlib import Path
from collections import deque
import math

ALPHAS    = [0.05, 0.1, 0.3]
GAMMAS    = [0.95, 0.99, 0.999]
EPS_SCHED = [                          # (eps0, eps_min, eps_decay)
    (1.0, 0.05, 0.990),
    (1.0, 0.05, 0.995),
    (1.0, 0.05, 0.999),
]

TRAIN_EPISODES = 5000
MASTER_SEED    = 42        # reproducible randomness
OUT_DIR        = Path("training/q-learning")
OUT_DIR.mkdir(exist_ok=True)


def train_qlearning(alpha: float, gamma: float,
                    eps0: float, eps_min: float, eps_decay: float,
                    episodes: int, seed: int
                   ) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    env = gym.make("Taxi-v3")
    env.reset(seed=seed)                       # set env RNG once
    rng = np.random.default_rng(seed)          # NumPy RNG matched

    nS, nA = env.observation_space.n, env.action_space.n
    q      = np.zeros((nS, nA))

    returns, steps, penalties, successes= [], [], [], []
    
    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep)
        done = False
        eps  = max(eps_min, eps0 * (eps_decay ** ep))
        success = False
        ep_ret = ep_steps = ep_pen = 0
        while not done:
            if rng.random() < eps:
                a = rng.integers(nA)
            else:
                a = int(np.argmax(q[s]))

            s2, r, term, trunc, _ = env.step(a)
            if term: success = True
            done = term or trunc
            
            # Q‑update
            q[s, a] += alpha * (r + gamma * q[s2].max() - q[s, a])

            ep_ret  += r
            ep_steps += 1
            ep_pen  += int(r == -10)
            s = s2

        # episode finished
        returns.append(ep_ret)
        steps.append(ep_steps)
        penalties.append(ep_pen)
        successes.append(success)
        

    env.close()
    return q, returns, steps, penalties,successes

def evaluate_policy(q: np.ndarray, episodes: int = 10, seed: int = 999) -> tuple[list[int], list[int], list[int]]:
    env = gym.make("Taxi-v3")
    env.reset(seed=seed)
    returns, steps, penalties, successes = [], [], [], []

    for ep in range(episodes):
        s, _ = env.reset(seed=seed+ep)
        done = False
        ep_ret = ep_steps = ep_pen = 0
        success = False
        while not done:
            a = int(np.argmax(q[s]))
            s, r, term, trunc, _ = env.step(a)
            done = term or trunc
            if done: success = True
            ep_ret  += r
            ep_steps += 1
            ep_pen  += int(r == -10)
        returns.append(ep_ret)
        steps.append(ep_steps)
        penalties.append(ep_pen)
        successes.append(success)

    env.close()
    return returns, steps, penalties, successes

def save_curves(
    curves: tuple[list[int], list[int], list[int], list[bool]],
    title: str,
    fname: str,
    window: int = 100
):
    ret, stp, pen, succ = curves
    episodes = np.arange(1, len(ret) + 1)
    stem = Path(fname).stem

    #Returns plot, with max highlighted
    plt.figure(figsize=(8,4))
    plt.plot(episodes, ret, label="Return")
    # highlight max
    max_idx = int(np.argmax(ret))
    max_ep  = max_idx + 1
    max_val = ret[max_idx]
    plt.scatter(
        max_ep, max_val,
        s=100,
        facecolors='none',
        edgecolors='red',
        linewidths=2,
        label=f"Max Return ({max_val})"
    )
    plt.title(f"{title}  |  Returns")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    p1 = OUT_DIR / f"{stem}_returns.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=120)
    plt.close()

    # 2) Steps plot
    plt.figure(figsize=(8,4))
    plt.plot(episodes, stp, label="Steps", color=None)
    plt.title(f"{title}  |  Steps")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.grid(alpha=0.3)
    p2 = OUT_DIR / f"{stem}_steps.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=120)
    plt.close()

    # 3) Penalties plot
    plt.figure(figsize=(8,4))
    plt.plot(episodes, pen, label="Penalties", color=None)
    plt.title(f"{title}  |  Penalties")
    plt.xlabel("Episode")
    plt.ylabel("Penalties")
    plt.grid(alpha=0.3)
    p3 = OUT_DIR / f"{stem}_penalties.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=120)
    plt.close()

    # 4) Success‐rate per block of episodes
    if len(succ) > 10:
        n_blocks = math.ceil(len(succ) / window)
        rates = []
        for i in range(n_blocks):
            start = i * window
            end   = min(start + window, len(succ))
            block = succ[start:end]
            rates.append(sum(block) / len(block))
        blocks = np.arange(1, n_blocks + 1)

        plt.figure(figsize=(8,4))
        plt.plot(blocks, rates, marker='o')
        plt.title(f"{title}  |  Success Rate (block={window})")
        plt.xlabel("Block #")
        plt.ylabel("Success Rate")
        plt.grid(alpha=0.3)
        p4 = OUT_DIR / f"{stem}_success_rate.png"
        plt.tight_layout()
        plt.savefig(p4, dpi=120)
        plt.close()
    else:
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, succ)
        plt.title(f'{stem}_success_rate.png')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(alpha=0.3)
        fname = os.path.join(OUT_DIR, f'{stem}_episode_length.png')
        plt.tight_layout()
        plt.savefig(fname, dpi=120)
        plt.close()



# 5. Grid search + plotting

def run_grid():
    start = time.time()
    grid = itertools.product(ALPHAS, GAMMAS, EPS_SCHED)

    for i, (alpha, gamma, (eps0, eps_min, eps_dec)) in enumerate(grid):
        cfg_tag   = f"a{alpha}_g{gamma}_e{eps_dec}".replace(".", "_")
        # … TRAIN with seed_train = MASTER_SEED + i …
        seed_train = MASTER_SEED + i
        q, tr_ret, tr_stp, tr_pen, tr_succ = train_qlearning(
            alpha, gamma, eps0, eps_min, eps_dec,
            TRAIN_EPISODES, seed=seed_train
        )
        save_curves((tr_ret, tr_stp, tr_pen, tr_succ),
                    f"TRAIN α={alpha} γ={gamma} εdec={eps_dec}",
                    f"train_{cfg_tag}.png")

        # EVAL with a *different* seed per model
        EVAL_EPISODES = 10
        seed_eval = MASTER_SEED + 100 + i
        ev_ret, ev_stp, ev_pen, ev_succ = evaluate_policy(
            q,
            episodes=EVAL_EPISODES,
            seed=seed_eval
        )
        save_curves((ev_ret, ev_stp, ev_pen, ev_succ),
                    f"EVAL α={alpha} γ={gamma} εdec={eps_dec}",
                    f"eval_{cfg_tag}.png")
        # ── print eval summary metrics ─────────────────────────────────
        avg_ret   = np.mean(ev_ret)
        avg_steps = np.mean(ev_stp)
        avg_pen   = np.mean(ev_pen)
        success_rate = (sum(ev_succ) / len(ev_succ)
                        if ev_succ else float('nan'))

        print(
            f"Eval Metrics [{cfg_tag}]: "
            f"Avg Return = {avg_ret:.2f}, "
            f"Avg Steps = {avg_steps:.2f}, "
            f"Avg Penalties = {avg_pen:.2f}, "
            f"Success Rate = {success_rate:.2%}"
        )
        

    print(f"All PNGs saved to {OUT_DIR}.  Done in {time.time()-start:,.1f}s")

if __name__ == "__main__":
    run_grid()

