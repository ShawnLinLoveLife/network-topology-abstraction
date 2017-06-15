#!/usr/bin/env python2

import networkx as nx
from networkx import connected_components
import matplotlib.pyplot as plt


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology

topology_name = "Aarnet"
filename = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/" + topology_name + ".graphml"
topo = nx.read_graphml(filename).to_undirected()
cc_topo = select_largest_cc(topo)

print("========Node========")
for u in cc_topo.nodes():
    print(cc_topo[u])

print("========Link========")
links_values = []
labels = []
node_labels = {}
edge_labels = {}

#for (u, v) in cc_topo.edges():
#    print(cc_topo.edge[u][v]['label'])

for (u, v) in cc_topo.edges():
    if 'label' not in cc_topo.edge[u][v].keys():
        links_values.append(4/4.0)
        edge_labels.setdefault((u, v), 'null')
        cc_topo.edge[u][v]['capability'] = 1000
        continue

    bw = str((cc_topo.edge[u][v]['label']))

    unit = 1
    if bw.count('Gbps') >= 1:
        unit = 10**9
    elif bw.count('Mbps') >= 1:
        unit = 10**6

    if bw.count('10'):
        cc_topo.edge[u][v]['capability'] = 10 * unit
    elif bw.count('2.5'):
        cc_topo.edge[u][v]['capability'] = 2.5 * unit
    elif bw.count('155'):
        cc_topo.edge[u][v]['capability'] = 155 * unit
    else:
        cc_topo.edge[u][v]['capability'] = 1000



for (u, v) in cc_topo.edges():
    print(cc_topo[u][v])

