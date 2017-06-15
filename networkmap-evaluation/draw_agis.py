#!/usr/bin/env python2

import networkx as nx
from networkx import connected_components
import matplotlib.pyplot as plt


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology

topology_name = "Agis"
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
for (u, v) in cc_topo.edges():
    if str(cc_topo.edge[u][v]['label']).count('45') >= 1:
        cc_topo.edge[u][v]['capacity'] = 45
    elif str(cc_topo.edge[u][v]['label']).count('155') >= 1:
        cc_topo.edge[u][v]['capacity'] = 155
    else:
        cc_topo.edge[u][v]['capacity'] = str('null')

for (u,v) in cc_topo.edges():
    print(cc_topo.edge[u][v])


for u in cc_topo.nodes():
    node_labels.setdefault(u, u)

print(edge_labels)



