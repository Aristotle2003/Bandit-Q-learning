from typing import Tuple, List
from graph import Graph

import networkx as nx
import matplotlib.pyplot as plt
import numpy.typing as npt

def draw_graph(
    graph: nx.DiGraph, 
    diameter: int,
    ax,
) -> None:
    pos = {(i, j): (i, -j) for i in range(diameter) for j in range(diameter)}
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=True,
        arrows=True,
        font_size=5,
        ax=ax,
    )

def highlight_shortest_path(
    graph: nx.DiGraph, 
    shortest_path: List,
    diameter: int,
    ax,
) -> None:
    edges = list(zip(shortest_path[:-1], shortest_path[1:]))
    pos = {(i, j): (i, -j) for i in range(diameter) for j in range(diameter)}
    nx.draw_networkx_edges(
        graph, 
        pos=pos, 
        edgelist=edges, 
        edge_color="green",
        width=2,
        arrows=True,
        ax=ax,
    )    

def highlight_path(
    graph: nx.DiGraph, 
    path: List, 
    diameter: int,
    ax,
) -> None:
    edges = list(zip(path[:-1], path[1:]))
    pos = {(i, j): (i, -j) for i in range(diameter) for j in range(diameter)}
    nx.draw_networkx_edges(
        graph, 
        pos=pos, 
        edgelist=edges, 
        edge_color="coral",
        width=2,
        arrows=True,
        ax=ax,
    )


def plot_graphs(
    graph: Graph,
    best_arms_found: list,
    values: list,
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    axs = axes.flatten()

    for i in range(len(axs)):
        draw_graph(graph.graph, graph.diameter, axs[i])
        if i == 0:
            highlight_shortest_path(graph.graph, graph.shortest_path(), graph.diameter, axs[i])
            axs[i].set_title("Optimal path")
        else:
            highlight_path(graph.graph, graph.arms[best_arms_found[i-1]], graph.diameter, axs[i])
            axs[i].set_title("Parameter value %1.2f" % values[i-1])

    fig.tight_layout()
    plt.show()


def plot_results(
    regret: npt.NDArray,
    percentage: npt.NDArray,
    values: list,
) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 6))
    axs = axes.flatten()
    ax1 = axs[0]
    ax2 = axs[1]

    for i in range(len(values)):
        ax1.plot(regret[i], label=values[i])
    ax1.set_title("Regret")
    ax1.set_xlabel("Time Steps")
    ax1.set_ylabel("Regret")

    for p in percentage:
        ax2.plot(p)
    ax2.set_title("Optimal Path Frequency")
    ax2.set_xlabel("Time Steps")
    ax2.set_ylabel("Percentage")

    fig.legend(loc=(0.1,0.8))
    fig.tight_layout()
    plt.show()