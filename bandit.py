from typing import Tuple
from graph import Graph

import numpy as np
import numpy.typing as npt


class Bandit:

    def __init__(
            self,
            graph: Graph,
            conditional_sigma: float,
            strategy: int,
            value: float,
            N: int
    ):
        self.graph = graph
        self.arms = self.graph.arms
        self.edges = self.graph.edges

        self.conditional_sigma = conditional_sigma
        self.strategy = strategy
        self.value = value
        self.N = N

        self.Qvalues = np.zeros(len(self.arms))
        self.arm_counts = np.zeros(len(self.arms))


    def simulate(self) -> Tuple[npt.NDArray, npt.NDArray]:
        regret = np.zeros(self.N)
        best_arm = self.graph.shortest_path_ind()
        best_arm_reward = self.pull_arm(best_arm)

        for t in range(self.N):
            if self.strategy == 0:  # ε-greedy
                chosen_arm = self.choose_arm_egreedy()
            elif self.strategy == 1:  # ε-decaying
                chosen_arm = self.choose_arm_edecay(t)
            elif self.strategy == 2:  # UCB
                chosen_arm = self.choose_arm_ucb(t)

            reward = self.pull_arm(chosen_arm)
            self.arm_counts[chosen_arm] += 1
            # Update Q-value using an unweighted moving average
            self.Qvalues[chosen_arm] += (reward - self.Qvalues[chosen_arm]) / self.arm_counts[chosen_arm]

            # Update regret
            actual_regret = best_arm_reward - reward
            regret[t] = actual_regret if chosen_arm != best_arm else 0
            return self.Qvalues, regret

    def choose_arm_egreedy(self) -> int:
        if np.random.rand() < self.value:  # self.value is used as ε
            return np.random.randint(0, len(self.Qvalues))  # Explore: random action
        else:
            return np.argmax(self.Qvalues)  # Exploit: best known action



    def choose_arm_edecay(self, t: int) -> int:
        K = len(self.Qvalues)  # Number of arms
        decayed_epsilon = min(1, self.value * K / (t + 1))  # self.value is used as the decay parameter c
        if np.random.rand() < decayed_epsilon:
            return np.random.randint(0, K)  # Explore: random action
        else:
            return np.argmax(self.Qvalues)  # Exploit: best known action



    def choose_arm_ucb(self, t: int) -> int:
        total_counts = np.sum(self.arm_counts)
        ucb_values = self.Qvalues + np.sqrt(2 * np.log(total_counts + 1) / (self.arm_counts + 1))
        return np.argmax(ucb_values)



    def pull_arm(
        self,
        idx: int,
    ) -> float:
        reward = 0
        for i in range(len(self.arms[idx]) - 1):
            mu_edge = self.edges[self.arms[idx][i]][self.arms[idx][i + 1]]["mu"]
            conditional_mean = np.log(mu_edge) - 0.5 * (self.conditional_sigma ** 2)
            reward -= np.exp(conditional_mean + self.conditional_sigma * np.random.randn())
        return reward


    def get_path_mean(
        self,
        idx: int,
    ) -> float:
        return -self.graph.all_path_means[idx]