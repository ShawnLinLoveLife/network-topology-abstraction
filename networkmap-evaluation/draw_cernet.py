#!/usr/bin/env python2

import networkx as nx
from networkx import connected_components
import matplotlib.pyplot as plt


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology

topology_name = "Cernet"
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
    if 'label' not in cc_topo.edge[u][v].values()[0].keys():
        links_values.append(4/4.0)
        edge_labels.setdefault((u, v), 'null')
        continue
    bw = str((cc_topo.edge[u][v].values()[0]['label']))
    edge_labels.setdefault((u, v), bw)
    if bw == '2M':
        links_values.append(1/4.0)
    elif bw == '155M':
        links_values.append(2/4.0)
    elif bw == '2.5G':
        links_values.append(3/4.0)
    else:
        links_values.append(4/4.0)

for u in cc_topo.nodes():
    node_labels.setdefault(u, u)

print(node_labels)
print(links_values)
pos = nx.spring_layout(cc_topo, scale=2)
print(pos)
nx.draw(cc_topo, pos)
nx.draw_networkx_edge_labels(cc_topo, pos, edge_labels=edge_labels)
values1 = plt.cm.Set3(links_values)
nx.draw_networkx_edges(cc_topo, pos, edge_color=values1, cmap=plt.get_cmap('Accent'), width=5)
#nx.draw_networkx_nodes(cc_topo, pos, label=node_labels)
nx.draw_networkx_labels(cc_topo, pos, labels=node_labels)
print(edge_labels)
#nx.draw_networkx(cc_topo, edge_color=values1, cmap=plt.get_cmap('Accent'), edge_labels=edge_labels)
plt.show()

