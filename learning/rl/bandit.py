import itertools
from collections import defaultdict
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


class Bandit:
    def __init__(self, n_arms: int = 10):
        self.n_arms = n_arms
        self.q_stars = np.random.normal(0, 1, n_arms)

    def __call__(self, arm_idx: int) -> float:
        # Random walk
        self.q_stars += np.random.normal(0, 0.01, self.n_arms)
        reward = np.random.normal(self.q_stars[arm_idx], 1)
        return reward


class EpsilonGreedyAgent:
    def __init__(self, n_arms: int = 10, eps: float = 0.1):
        self.n_arms = n_arms
        self.eps = eps
        self.counts = np.zeros(n_arms)
        self.q_hats = np.zeros(n_arms)

    def select_action(self) -> int:
        eps_prob = np.random.rand()
        if eps_prob < self.eps:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.q_hats)

    def update(self, arm_idx: int, reward: float):
        raise NotImplementedError("Not implemented")


class NonStationaryAgent(EpsilonGreedyAgent):
    def __init__(self, n_arms: int = 10, eps: float = 0.1):
        super().__init__(n_arms, eps)

    def update(self, arm_idx: int, reward: float):
        self.counts[arm_idx] += 1
        self.q_hats[arm_idx] += (1 / self.counts[arm_idx]) * (
            reward - self.q_hats[arm_idx]
        )


class StationaryAgent(EpsilonGreedyAgent):
    def __init__(self, n_arms: int = 10, eps: float = 0.1, alpha: float = 0.1):
        super().__init__(n_arms, eps)
        self.alpha = alpha

    def update(self, arm_idx: int, reward: float):
        self.q_hats[arm_idx] += self.alpha * (reward - self.q_hats[arm_idx])


def run_simple_non_stationary_bandit(steps: int = 1000, eps: float = 0.1):
    mab = Bandit()
    agent = NonStationaryAgent(eps=eps)

    rewards = []

    for i in range(steps):
        arm_idx = agent.select_action()
        reward = mab(arm_idx)
        agent.update(arm_idx, reward)
        rewards.append(reward)

    is_optimal_action = np.argmax(mab.q_stars) == np.argmax(agent.q_hats)
    return rewards, is_optimal_action


def run_simple_stationary_bandit(steps: int = 1000, alpha: float = 0.1):
    mab = Bandit()
    agent = StationaryAgent(alpha=alpha)

    rewards = []

    for i in range(steps):
        arm_idx = agent.select_action()
        reward = mab(arm_idx)
        agent.update(arm_idx, reward)
        rewards.append(reward)

    is_optimal_action = np.argmax(mab.q_stars) == np.argmax(agent.q_hats)
    return rewards, is_optimal_action


def run_experiment_batch_nonstationary(args):
    """Run a batch of trials for non-stationary bandit experiments"""
    step, eps, alpha, trials = args
    results = []
    for _ in range(trials):
        rewards, is_optimal_action = run_simple_non_stationary_bandit(
            steps=step, eps=eps
        )
        results.append((rewards, is_optimal_action))
    return (step, eps, alpha), results


def run_experiment_batch_stationary(args):
    """Run a batch of trials for stationary bandit experiments"""
    step, eps, alpha, trials = args
    results = []
    for _ in range(trials):
        rewards, is_optimal_action = run_simple_stationary_bandit(
            steps=step, alpha=alpha
        )
        results.append((rewards, is_optimal_action))
    return (step, eps, alpha), results


if __name__ == "__main__":
    # experiment grid - OPTIMIZED FOR SPEED
    steps = np.logspace(1, 3, 20).astype(
        int
    )  # 20 points from 10 to 1000 (logarithmic spacing)
    eps = [0, 0.01, 0.1]
    alpha = [0.1]

    # Cartesian
    experiments = list(itertools.product(steps, eps, alpha))
    trials = 100  # Reduced from 2000 to 100 for much faster execution

    print(f"Total experiments per agent type: {len(experiments)}")
    print(f"Total simulations per agent type: {len(experiments) * trials:,}")
    print(f"Running both stationary and non-stationary agents...")

    # Run non-stationary experiments
    print("Running non-stationary experiments in parallel...")
    experiment_args = [(step, eps, alpha, trials) for step, eps, alpha in experiments]

    with Pool(processes=cpu_count()) as pool:
        nonstationary_results = list(
            tqdm(
                pool.imap(run_experiment_batch_nonstationary, experiment_args),
                total=len(experiment_args),
                desc="Non-stationary experiments",
            )
        )

    # Run stationary experiments
    print("Running stationary experiments in parallel...")

    with Pool(processes=cpu_count()) as pool:
        stationary_results = list(
            tqdm(
                pool.imap(run_experiment_batch_stationary, experiment_args),
                total=len(experiment_args),
                desc="Stationary experiments",
            )
        )

    # Collect results
    nonstationary_dict = defaultdict(list)
    stationary_dict = defaultdict(list)

    for (step, eps, alpha), trial_results in nonstationary_results:
        nonstationary_dict[(step, eps, alpha)] = trial_results

    for (step, eps, alpha), trial_results in stationary_results:
        stationary_dict[(step, eps, alpha)] = trial_results

    # Process results for plotting - Non-stationary
    nonstationary_eps_results = defaultdict(
        lambda: {"steps": [], "avg_rewards": [], "optimal_action_pct": []}
    )

    for (step, eps, alpha), trial_results in nonstationary_dict.items():
        # Calculate average reward across all trials (final reward of each trial)
        final_rewards = [rewards[-1] for rewards, _ in trial_results]
        avg_final_reward = np.mean(final_rewards)

        # Calculate percentage of trials that ended with optimal action
        optimal_actions = [is_optimal for _, is_optimal in trial_results]
        optimal_action_percentage = np.mean(optimal_actions) * 100

        nonstationary_eps_results[eps]["steps"].append(step)
        nonstationary_eps_results[eps]["avg_rewards"].append(avg_final_reward)
        nonstationary_eps_results[eps]["optimal_action_pct"].append(
            optimal_action_percentage
        )

    # Process results for plotting - Stationary
    stationary_eps_results = defaultdict(
        lambda: {"steps": [], "avg_rewards": [], "optimal_action_pct": []}
    )

    for (step, eps, alpha), trial_results in stationary_dict.items():
        # Calculate average reward across all trials (final reward of each trial)
        final_rewards = [rewards[-1] for rewards, _ in trial_results]
        avg_final_reward = np.mean(final_rewards)

        # Calculate percentage of trials that ended with optimal action
        optimal_actions = [is_optimal for _, is_optimal in trial_results]
        optimal_action_percentage = np.mean(optimal_actions) * 100

        stationary_eps_results[eps]["steps"].append(step)
        stationary_eps_results[eps]["avg_rewards"].append(avg_final_reward)
        stationary_eps_results[eps]["optimal_action_pct"].append(
            optimal_action_percentage
        )

    # Plot 1: Average Final Reward vs Steps - Non-Stationary
    plt.figure(figsize=(10, 6))
    for eps in sorted(nonstationary_eps_results.keys()):
        data = nonstationary_eps_results[eps]
        # Sort by steps for proper line plotting
        sorted_indices = np.argsort(data["steps"])
        steps_sorted = np.array(data["steps"])[sorted_indices]
        rewards_sorted = np.array(data["avg_rewards"])[sorted_indices]
        plt.plot(
            steps_sorted, rewards_sorted, label=f"ε = {eps}", marker="o", markersize=2
        )

    plt.xlabel("Steps")
    plt.ylabel("Average Final Reward")
    plt.title("Average Final Reward vs Steps (Non-Stationary Bandit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rl/nonstationary_bandit_avg_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 2: Optimal Action Percentage vs Steps - Non-Stationary
    plt.figure(figsize=(10, 6))
    for eps in sorted(nonstationary_eps_results.keys()):
        data = nonstationary_eps_results[eps]
        # Sort by steps for proper line plotting
        sorted_indices = np.argsort(data["steps"])
        steps_sorted = np.array(data["steps"])[sorted_indices]
        optimal_pct_sorted = np.array(data["optimal_action_pct"])[sorted_indices]
        plt.plot(
            steps_sorted,
            optimal_pct_sorted,
            label=f"ε = {eps}",
            marker="o",
            markersize=2,
        )

    plt.xlabel("Steps")
    plt.ylabel("Optimal Action Percentage (%)")
    plt.title(
        "Percentage of Trials with Optimal Action Selected (Non-Stationary Bandit)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig(
        "rl/nonstationary_bandit_optimal_action_pct.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot 3: Average Final Reward vs Steps - Stationary
    plt.figure(figsize=(10, 6))
    for eps in sorted(stationary_eps_results.keys()):
        data = stationary_eps_results[eps]
        # Sort by steps for proper line plotting
        sorted_indices = np.argsort(data["steps"])
        steps_sorted = np.array(data["steps"])[sorted_indices]
        rewards_sorted = np.array(data["avg_rewards"])[sorted_indices]
        plt.plot(
            steps_sorted, rewards_sorted, label=f"ε = {eps}", marker="o", markersize=2
        )

    plt.xlabel("Steps")
    plt.ylabel("Average Final Reward")
    plt.title("Average Final Reward vs Steps (Stationary Bandit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("rl/stationary_bandit_avg_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Plot 4: Optimal Action Percentage vs Steps - Stationary
    plt.figure(figsize=(10, 6))
    for eps in sorted(stationary_eps_results.keys()):
        data = stationary_eps_results[eps]
        # Sort by steps for proper line plotting
        sorted_indices = np.argsort(data["steps"])
        steps_sorted = np.array(data["steps"])[sorted_indices]
        optimal_pct_sorted = np.array(data["optimal_action_pct"])[sorted_indices]
        plt.plot(
            steps_sorted,
            optimal_pct_sorted,
            label=f"ε = {eps}",
            marker="o",
            markersize=2,
        )

    plt.xlabel("Steps")
    plt.ylabel("Optimal Action Percentage (%)")
    plt.title("Percentage of Trials with Optimal Action Selected (Stationary Bandit)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)
    plt.savefig(
        "rl/stationary_bandit_optimal_action_pct.png", dpi=300, bbox_inches="tight"
    )
    plt.close()
