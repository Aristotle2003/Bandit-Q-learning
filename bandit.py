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


    def simulate(self):
        regret = np.zeros(self.N)
        best_arm = self.graph.shortest_path_ind()
        best_arm_reward = self.pull_arm(best_arm)

        for t in range(self.N):
            if self.strategy == 0:  # ε-Greedy
                chosen_arm = self.choose_arm_egreedy()
            elif self.strategy == 1:  # ε-Decaying
                chosen_arm = self.choose_arm_edecay(t)
            else:  # UCB
                chosen_arm = self.choose_arm_ucb(t)

            reward = self.pull_arm(chosen_arm)
            self.arm_counts[chosen_arm] += 1
            # Update Q-value using incremental formula
            self.Qvalues[chosen_arm] += (reward - self.Qvalues[chosen_arm]) / self.arm_counts[chosen_arm]

            # Calculate regret
            if chosen_arm != best_arm:
                regret[t] = best_arm_reward - reward
            else:
                regret[t] = 0

        return self.Qvalues, regret

    def choose_arm_egreedy(self):
        if np.random.rand() < self.value:  # Explore
            return np.random.choice(len(self.arms))
        else:  # Exploit
            return np.argmax(self.Qvalues)

    def choose_arm_edecay(self, t):
        epsilon = min(1, self.value * len(self.arms) / (t + 1))
        if np.random.rand() < epsilon:  # Explore with decaying epsilon
            return np.random.choice(len(self.arms))
        else:  # Exploit
            return np.argmax(self.Qvalues)

    def choose_arm_ucb(self, t):
        exploration = np.sqrt(np.log(t+1) / (self.arm_counts+1))
        ucb_values = self.Qvalues + self.value * exploration
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