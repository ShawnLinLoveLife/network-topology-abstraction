#!/usr/bin/env python

import topology as tp
import networkx as nx
import drawtopology as dt
import itertools
import numpy as np


def gen_girvan_newman_connected_components(topo, K):
    ccs = []
    central = sorted(nx.edge_betweenness_centrality(topo).items(), key=lambda d: d[1], reverse=True)
    while len(central) != 0 and len(sorted(nx.connected_components(topo), key=len, reverse=True)) <= K:
        ccs = list(nx.connected_components(topo))
        topo.remove_edge(central[0][0][0], central[0][0][1])
        central = sorted(nx.edge_betweenness_centrality(topo).items(), key=lambda d: d[1], reverse=True)
    return ccs


def to_girvan_newman_networkmap(ccs):
    pid_number = len(ccs)
    pids = []
    for i in range(pid_number):
        pids.append(list(ccs[i]))
    return tuple(pids)


if __name__ == "__main__":
    topology_name = "Cernet"
    filename = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/"+topology_name+".graphml"
    topo = tp.process_topology(filename)
    CC_topo = dt.select_largest_cc(topo)
    origin_CC_topo = dt.select_largest_cc(tp.process_topology(filename))
    print(CC_topo)
    print("Start Girvan Newman")
    ccs = gen_girvan_newman_connected_components(CC_topo, 17)
    network_map = to_girvan_newman_networkmap(ccs)
    print(network_map)
    cost_map = dt.default_cost_map(origin_CC_topo, network_map)
    print(cost_map)
    host_list = []
    for node in origin_CC_topo.nodes():
        host_list.append(node)
    print(host_list)
    print("Result")
    g_n_spl_mat = dt.gen_spl_mat_from_maps(network_map, cost_map, host_list)
    origin_spl_mat = dt.gen_spl_mat_from_graph(origin_CC_topo, host_list)
    dist = (np.absolute(origin_spl_mat-g_n_spl_mat)).sum()/len(host_list)/len(host_list)
    print(dist)
    dt.print_graph(origin_CC_topo, network_map[0], network_map[1], topology_name, "girvan_newman", times=0, dist=dist)