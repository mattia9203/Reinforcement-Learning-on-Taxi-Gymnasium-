import gymnasium as gym
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from collections import deque, namedtuple
from DQN import DQN
import matplotlib.pyplot as plt
import os
import itertools
import math

SAVE_DIR = 'training/DQN/graphics'

# Replay Buffer
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curves(episode_rewards, episode_steps, episode_penalties, succ,
                eps_per_episode=None, episode=None, title_prefix='', window = 100):
    """Plot and save reward, steps, penalties and  epsilon curves."""
    episodes = np.arange(1, len(episode_rewards) + 1)
    if episode: title_prefix = title_prefix + f'_{episode}_episode'
    print(title_prefix)

    # Rewards and Epsilon
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_rewards, label='Reward')
    if eps_per_episode is not None:
        plt.plot(episodes, eps_per_episode, label='Epsilon')
    plt.title(f'{title_prefix} Rewards and Epsilon')
    plt.xlabel('Episode')
    plt.ylabel('Reward / Epsilon')
    plt.legend()
    fname = os.path.join(SAVE_DIR, f'{title_prefix}_rewards_epsilon.png')
    plt.savefig(fname)
    plt.close()

    # Episode Length
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_steps)
    plt.title(f'{title_prefix} Episode Length')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    fname = os.path.join(SAVE_DIR, f'{title_prefix}_episode_length.png')
    plt.savefig(fname)
    plt.close()

    # Episode Penalties
    plt.figure(figsize=(8, 4))
    plt.plot(episodes, episode_penalties)
    plt.title(f'{title_prefix} Episode Penalties')
    plt.xlabel('Episode')
    plt.ylabel('Penalties')
    fname = os.path.join(SAVE_DIR, f'{title_prefix}_episode_penalties.png')
    plt.savefig(fname)
    plt.close()

    if len(succ) > 100:
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
        plt.title(f"{title_prefix}  |  Success Rate (block={window})")
        plt.xlabel("Block #")
        plt.ylabel("Success Rate")
        plt.grid(alpha=0.3)
        fname = os.path.join(SAVE_DIR, f'{title_prefix}_success_rate.png')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
    else:
        plt.figure(figsize=(8, 4))
        plt.plot(episodes, succ, marker='o')
        plt.title(f'{title_prefix}_success_rate.png')
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.grid(alpha=0.3)
        fname = os.path.join(SAVE_DIR, f'{title_prefix}_success_rate.png')
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()


def evaluate_model(policy_net: torch.nn.Module, env_name: str, device: torch.device,
                   n_episodes: int = 10, seed: int = 1000):
    """
    Evaluate the policy_net on env_name for n_episodes with a greedy policy.
    Returns lists of (returns, steps, penalties, successes).
    """
    env = gym.make(env_name)
    env.reset(seed=seed)
    returns, steps, penalties, successes = [], [], [], []
    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed + ep)
        total_return = 0
        total_steps = 0
        total_penalties = 0
        success = False
        done = False
        while not done:
            with torch.no_grad():
                st = torch.tensor([state], dtype=torch.long).to(device)
                action = policy_net(st).argmax(dim=1).item()
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
            total_steps += 1
            if reward == -10:
                total_penalties += 1
            if reward > 0:
                success = True
            state = next_state
        returns.append(total_return)
        steps.append(total_steps)
        penalties.append(total_penalties)
        successes.append(success)
    env.close()
    return returns, steps, penalties, successes


def train(env_name, seed, lr, batch_size, target_update_freq,
          epsilon_type='decay', eps_start=1.0, eps_end=0.01,
          num_episodes=10000, capacity=10000, ddqn=False):
    """Train DQN with specified epsilon schedule and return metrics and best model path."""
    # Reproducibility
    set_seed(seed)

    # Environment setup
    env = gym.make(env_name)
    state, _ = env.reset(seed=seed)  
    env.action_space.seed(seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Networks
    policy_net = DQN(outputs=n_actions).to(device)
    target_net = DQN(outputs=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Replay buffer
    class ReplayBuffer:
        def __init__(self, capacity):
            self.buffer = deque(maxlen=capacity)
        def push(self, *args):
            self.buffer.append(Transition(*args))
        def sample(self, batch_size):
            return random.sample(self.buffer, batch_size)
        def __len__(self):
            return len(self.buffer)
    replay_buffer = ReplayBuffer(capacity)

    # Metrics and tracking
    episode_rewards, episode_steps, episode_penalties, episode_successes, eps_history = [], [], [], [], []
    best_eval_avg = -float('inf')
    best_eval_model_path = 'training/DQN/models'
    steps_done = 0
    eval_interval = 100
    eval_episodes = 10
    gamma = 0.99
    max_steps     = 200    
    total_steps   = num_episodes * max_steps
    if epsilon_type == 'decay_1':
        decay_factor  = (eps_end / eps_start) ** (1.0 / total_steps)
    if epsilon_type == 'decay_2':
        decay_factor  = (eps_end / eps_start) ** (1.0 / num_episodes*100)
    eps = eps_start
    # Training loop
    for episode in range(1, num_episodes + 1):
        state, _ = env.reset(seed=seed+episode)
        total_reward, steps_taken, penalties = 0, 0, 0
        done = False
        success = False
        eps_history.append(eps)
        while not done:
            # Epsilon-greedy action
            # Compute epsilon
            if epsilon_type == 'fixed_0.1':
                eps = 0.1
            if epsilon_type == 'decay_1':
                decay_factor  = (eps_end / eps_start) ** (1.0 / num_episodes)
                eps = max(eps_end, eps_start * (decay_factor ** episode))
            if epsilon_type == 'decay_2':
                decay_factor  = (eps_end / eps_start) ** (1.0 / num_episodes/2)
                eps = max(eps_end, eps_start * (decay_factor ** episode))

            if random.random() < eps:
                action = env.action_space.sample()
            else:
                with torch.no_grad():
                    st = torch.tensor([state], dtype=torch.long).to(device)
                    q_vals = policy_net(st)
                    action = q_vals.argmax(dim=1).item()

            # Environment step
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Metrics per step
            steps_taken += 1
            if reward == -10:
                penalties += 1
            total_reward += reward
            if reward == 20: success = True

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            steps_done += 1

            # Learning
            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.sample(batch_size)
                batch = Transition(*zip(*transitions))
                state_batch = torch.tensor(batch.state, dtype=torch.long).to(device)
                action_batch = torch.tensor(batch.action).unsqueeze(1).to(device)
                reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
                next_state_batch = torch.tensor(batch.next_state, dtype=torch.long).to(device)
                done_batch = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1).to(device)

                # Q-values and targets
                q_values = policy_net(state_batch).gather(1, action_batch)
                if (not ddqn):
                    with torch.no_grad():
                        next_q = target_net(next_state_batch).max(1)[0].unsqueeze(1)
                        q_targets = reward_batch + gamma * next_q * (1 - done_batch)
                else:
                    with torch.no_grad():
                        # Use policy_net to select the best next action (action selection)
                        best_next_actions = policy_net(next_state_batch).argmax(1, keepdim=True)

                        # Use target_net to evaluate that action (action evaluation)
                        next_q_values = target_net(next_state_batch).gather(1, best_next_actions)

                        q_targets = reward_batch + gamma * next_q_values * (1 - done_batch)

                # Optimize
                loss = F.smooth_l1_loss(q_values, q_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # Update target network
            if steps_done % target_update_freq == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # End of episode logging
        episode_rewards.append(total_reward)
        episode_steps.append(steps_taken)
        episode_penalties.append(penalties)
        episode_successes.append(success)
        # Periodic evaluation and checkpointing
        if episode % eval_interval == 0:
            returns, steps_e, penalties_e, succ_e = evaluate_model(policy_net, env_name, device,
                                                                    eval_episodes, seed + episode)
            avg_ret = np.mean(returns)
            print(f"[Eval] Episode {episode}: avg_return={avg_ret:.2f}")
            if avg_ret > best_eval_avg:
                plot_curves(returns, steps_e, penalties_e, succ_e, episode=episode, title_prefix='eval')
                best_eval_avg = avg_ret
                best_eval_model_path = os.path.join('training/DQN/models',
                    f"dqn_best_eval_seed{seed}_lr{lr}_bs{batch_size}_tgt{target_update_freq}_{epsilon_type}_ep{episode}.pth")
                torch.save(policy_net.state_dict(), best_eval_model_path)
                print(f"  New best eval model saved: {best_eval_model_path}")

        if episode % 100 == 0:
            print(f"Seed {seed} | Ep {episode} | Rew {total_reward:.2f} | Eps {eps:.3f}")
    env.close()
    
    return (episode_rewards, episode_steps, episode_penalties,
            episode_successes, eps_history,
            best_eval_avg, best_eval_model_path)

        

def main():
    env_name = 'Taxi-v3'
    seeds = [42]
    lr_list = [1e-3, 1e-4, 1e-5]
    batch_list = [64, 128]
    target_updates = [1000, 2000]
    epsilon_types = ['decay_1', 'decay_2']

    results = []
    for seed, lr, batch_size, tgt_upd, eps_type in itertools.product(
            seeds, lr_list, batch_list, target_updates, epsilon_types):
        print(f"\nTraining: seed={seed}, lr={lr}, batch={batch_size}, tgt_upd={tgt_upd}, eps={eps_type}")
        ep_rew, ep_steps, ep_pen, ep_succ, eps_ep, best_rew, best_model = train(
            env_name, seed, lr, batch_size, tgt_upd, epsilon_type=eps_type, ddqn=True)
        title = f"S{seed}_lr{lr}_bs{batch_size}_tgt{tgt_upd}_eps{eps_type}"
        plot_curves(ep_rew, ep_steps, ep_pen, ep_succ, eps_per_episode=eps_ep, title_prefix=title)
        results.append((seed, lr, batch_size, tgt_upd, eps_type, best_rew, best_model))

    print("\nGrid Search Results:")
    for seed, lr, bs, tu, eps_type, best_rew, model_path in results:
        print(f"seed={seed}, lr={lr}, batch={bs}, tgt_upd={tu}, eps={eps_type} "
              f"→ best_reward={best_rew:.2f}, model={model_path}")


if __name__ == '__main__':
    main()
