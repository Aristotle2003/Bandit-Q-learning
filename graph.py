from collections import defaultdict
from typing import Dict, Any

import networkx as nx
import numpy as np


class Graph:

    def __init__(
        self, 
        params: Dict[str, Any] = None,
    ):
        self.edges = defaultdict(dict)
        self.diameter = params["diameter"]
        self.graph = self.__create_network(params["prior_mu"], params["prior_sigma"], params["conditional_sigma"])
        self.arms = self.__create_arms()
        self.all_path_means = self.__all_path_means()


    def __create_network(self, prior_mu, prior_sigma, conditional_sigma):
        G = nx.DiGraph()
        for i in range(self.diameter):
            for j in range(self.diameter):
                if i < self.diameter - 1:
                    out_v, in_v = (i, j), (i + 1, j)
                    mu_edge = np.exp(prior_mu + prior_sigma * np.random.randn())
                    G.add_edge(out_v, in_v, weight=mu_edge)
                    self.edges[out_v][in_v] = {"mu": mu_edge, "sigma": conditional_sigma}
                if j < self.diameter - 1:
                    out_v, in_v = (i, j), (i, j + 1)
                    mu_edge = np.exp(prior_mu + prior_sigma * np.random.randn())
                    G.add_edge(out_v, in_v, weight=mu_edge)
                    self.edges[out_v][in_v] = {"mu": mu_edge, "sigma": conditional_sigma}
        return G
    

    def __all_path_means(self):
        path_lens_sample = defaultdict(float)
        for idx, path in self.arms.items():
            for i in range(len(path) - 1):
                path_lens_sample[idx] += self.edges[path[i]][path[i + 1]]["mu"]
        return path_lens_sample
    

    def __update_path_means(self):
        for idx, path in self.arms.items():
            self.all_path_means[idx] = 0
            for i in range(len(path) - 1):
                self.all_path_means[idx] += self.edges[path[i]][path[i + 1]]["mu"]
        return self.all_path_means
    
    
    def __create_arms(self):
        arms = defaultdict(list)
        paths = list(nx.all_simple_paths(self.graph, (0, 0), (self.diameter - 1, self.diameter - 1)))
        for i, path in enumerate(sorted(paths)):
            arms[i] = path
        return arms
    
    
    def shortest_path(self):
        return nx.shortest_path(self.graph, (0, 0), (self.diameter - 1, self.diameter - 1), weight="weight")
    
    
    def shortest_path_len(self):
        return nx.shortest_path_length(self.graph, (0, 0), (self.diameter - 1, self.diameter - 1), weight="weight")


    def shortest_path_ind(self):
        return min(self.all_path_means, key=lambda k: self.all_path_means[k])
    

    def update_edges(self, path):
        for out_v, in_v, post_mu_edge, post_sigma_edge in path:
            self.edges[out_v][in_v]["mu"] = post_mu_edge
            self.edges[out_v][in_v]["sigma"] = post_sigma_edge
            self.graph.add_edge(out_v, in_v, weight=post_mu_edge)

        _ = self.__update_path_means()  

        return None