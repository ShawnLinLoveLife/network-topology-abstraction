#!/usr/bin/env python

import networkx as nx
from networkx import connected_components
import matplotlib.pyplot as plt

anchor = []


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology


topology_name = "Cernet"
filename = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/" + topology_name + ".graphml"
topo = nx.read_graphml(filename).to_undirected()
cc_topo = select_largest_cc(topo)


for (u, v) in cc_topo.edges():
    if 'label' not in cc_topo.edge[u][v].values()[0].keys():
        continue
    bw = str((cc_topo.edge[u][v].values()[0]['label']))
    if bw == '2.5G':
        if u not in anchor:
            anchor.append(u)
        if v not in anchor:
            anchor.append(v)

print(anchor)
# ['n29', 'n15', 'n22', 'n4', 'n26', 'n25']