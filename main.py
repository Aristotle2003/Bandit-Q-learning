from bandit import Bandit
from graph import Graph
from utils import *

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(
        prog="COMSW4701HW3b",
        description="Multi-Armed Bandit for Online SPP",
    )
    parser.add_argument(
        "-w", default=2, type=int, help="width of the binomial bridge",
    )
    parser.add_argument(
        "-s", default=0, type=int, choices=range(0,3), help="bandit strategy",
    )
    parser.add_argument(
        "-pm", default=-0.5, type=float, help="prior mu for weight generation",
    )
    parser.add_argument(
        "-ps", default=1, type=float, help="prior sigma for weight generation",
    )
    parser.add_argument(
        "-cs", default=1, type=float, help="conditional sigma for weight generation",
    )
    parser.add_argument(
        "-m", default=100, type=int, help="number of simulation runs",
    )
    parser.add_argument(
        "-n", default=10000, type=int, help="number of time steps per simulation",
    )
    args = parser.parse_args()

    params = {"diameter": args.w, "prior_mu": args.pm, "prior_sigma": args.ps, "conditional_sigma": args.cs}
    graph = Graph(params)
    bandit = Bandit(graph, args.cs, args.s, 0, args.n)
    best_arms_found = []
    regrets_over_time = []
    proportions_over_time = []

    print("=" * 40)
    if args.s == 0:
        print("Testing Epsilon-Greedy...\n")
        values = [0.1, 0.2, 0.5]
    elif args.s == 1:
        print("Testing Epsilon-Decaying...\n")
        values = [1, 5, 10]
    else:
        print("Testing UCB...\n")
        values = [1.0, 2.0, 5.0]

    for v in values:
        bandit.value = v
        best_arm = []
        regrets = np.zeros(args.n)
        best_arm_counts = np.zeros(args.n)

        for _ in range(args.m):
            Q, regret = bandit.simulate()
            best_arm.append(np.argmax(Q))
            regrets += regret

            temp = np.zeros(args.n)
            temp[np.where(regret == 0)[0]] = 1
            best_arm_counts += temp

        best_arms_found.append(max(set(best_arm), key=best_arm.count))
        regrets_over_time.append(np.cumsum(regrets) / args.m)
        proportions_over_time.append(np.cumsum(best_arm_counts) / np.arange(1, args.n+1))

    print("=" * 40)
    print("Plotting Results...\n")
    plot_graphs(graph, best_arms_found, values)
    plot_results(regrets_over_time, proportions_over_time, values)

    print("Done!")
    print("=" * 40)


if __name__ == "__main__":
    main()