#!/usr/bin/env python

from functools import reduce
import sys
import re
import fnss
from networkx import connected_components, all_pairs_shortest_path, DiGraph
from networkx import read_graphml, write_graphml
import networkx as nx
import matplotlib.pyplot as plt


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology

rocketfuel_bw_table = [100000, 40000, 20000, 10000, 5000, 2000] # Mbps
topologyzoo_bw_table = [100, 40, 10]


def rocketfuel_bandwidth_func(x, y, node_x, node_y):
    rx, ry = node_x['r'], node_y['r']
    if rx == ry:
        return rocketfuel_bw_table[rx]
    else:
        rmax = int((rx+ry*2)/3)
        return rocketfuel_bw_table[rmax]


def ensure_internal_info(g):
    has_internal = reduce(lambda x,y: x or 'Internal' in g.node[y], g.nodes(), False)
    if not has_internal:
        apsp = all_pairs_shortest_path(g)
        dmax = {n: max(map(lambda p: len(apsp[n][p]), apsp[n])) for n in g.nodes()}
        dall = [dmax[n] for n in g.nodes()]
        dave = sum(dall) / len(dall)
        for n in g.nodes():
            g.node[n]['Internal'] = True if dmax[n] < dave else False

    for n in g.nodes():
        g.node[n]['type'] = 'internal' if g.node[n]['Internal'] else 'external'


def ensure_bandwidth_info(g):
    total = [0, 0, 0]
    cnt = [0, 0, 0]
    for (u, v) in g.edges():
        type = (g.node[u]['type'] != 'internal') + (g.node[v]['type'] != 'internal')
        g[u][v]['type'] = type
        if 'label' not in g.edge[u][v]:
            g[u][v]['label'] = str(topologyzoo_bw_table[type]) + " Mbps"
        label = g[u][v]['label']
        m = re.search('([0-9]+) ([GMK])b', label)
        if m is None:
            continue
        unit = 10**9 if 'G' == m.group(2) else 10**6 if 'M' else 10**3
        bw = int(m.group(1)) * unit
        g[u][v]['bandwidth'] = bw
        total[type] += bw
        cnt[type] += 1

    if reduce(lambda x,y: x or y == 0, cnt, False):
        ave = topologyzoo_bw_table
    else:
        ave = [total[i] / cnt[i] for i in range(3)]
    for u, v in g.edges():
        if 'bandwidth' not in g[u][v]:
            g[u][v]['bandwidth'] = ave[g[u][v]['type']]


def unify_graph_attr(topology, topology_type='rocketfuel'):
    if 'rocketfuel' == topology_type:
        excluded_func = lambda n: n['type'] == 'internal'
        bw_func = rocketfuel_bandwidth_func
    else:
        ensure_internal_info(topology)
        ensure_bandwidth_info(topology)
        excluded_func = lambda n: n['type'] == 'internal'
        bw_func = lambda x,y,nx,ny: topology.edge[x][y]['bandwidth']

    for nid in topology.nodes():
        n = topology.node[nid]
        topology.node[nid]['excluded'] = excluded_func(n)

    for x, y in topology.edges():
        nx, ny = topology.node[x], topology.node[y]
        topology.edge[x][y]['capacity'] = bw_func(x, y, nx, ny)
        topology.edge[x][y]['load'] = 0
    topology.graph['capacity_unit'] = "Mbps"

    return topology


def load_topology(filename, topology_type='rocketfuel'):
    if 'topologyzoo' == topology_type:
        topology = fnss.parse_topology_zoo(filename).to_undirected()
    elif 'rocketfuel':
        topology = fnss.parse_rocketfuel_isp_map(filename).to_undirected()
    else:
        topology = None
        print("Unsupported file format!")
        sys.exit(-1)
    return select_largest_cc(topology)


def dump_topology(g, outfile):
    print(len(g.nodes()))
    print(len(g.edges()))
    candidates = list(filter(lambda n: g.node[n]['excluded'] == False, g.nodes()))
    print(len(candidates))
    print(' '.join(map(str, candidates)))

    for (u, v) in g.edges():
        print(u, v, g.edge[u][v]['capacity'])


def dump_raw_topology(g, filename):
    ng = DiGraph()
    for (u, v) in g.edges():
        ng.add_edge(u, v, capacity=g.edge[u][v]['capacity'])
        ng.add_edge(v, u, capacity=g.edge[u][v]['capacity'])

    for u in g.nodes():
        ng.node[u]['internal'] = not g.node[u]['excluded']

    write_graphml(ng, filename)


def load_raw_topology(filename):
    return read_graphml(filename)


def process_topology(filename, outfile=sys.stderr):
    ext = filename.split('.')[-1]
    print("Loading topology...")
    topo_type = 'rocketfuel' if ext == 'cch' else 'topologyzoo'
    g = load_topology(filename, topo_type)
    g = unify_graph_attr(g, topo_type)
    print("Loading topology...Done")

    dump_topology(g, outfile)

    return g

if __name__ == "__main__":
    #filename = sys.argv[1]
    filename = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/Chinanet.graphml"
    g = process_topology(filename)
    print("show node")
    for node in g.nodes():
        print(g.node[node])

    for (u,v) in g.edges():
        print(g.edge[u][v])

    # Print result
    # Deal with nodes
    nodes_values = []
    nodes_labels = {}
    for node in g.nodes():
        nodes_labels.setdefault(u, u)
        if g.node[node]['type'] == "internal":
            nodes_values.append(1/10.0)
        else:
            nodes_values.append(2/10.0)
    set3_nodes_values = plt.cm.Set3(nodes_values)

    # Deal with links
    edges_values = []
    edges_labels = {}
    for (u,v) in g.edges():
        print(g.edge[u][v])
        if g.edge[u][v]['capacity'] == 10000000:
            edges_values.append(2/3.0)
            edges_labels.setdefault((u, v), '2ms')
        elif g.edge[u][v]['capacity'] == 40000000:
            edges_values.append(3/3.0)
            edges_labels.setdefault((u, v), '10ms')
        else:
            edges_values.append(1/3.0)
            edges_labels.setdefault((u, v), '20ms')
    set3_edges_values = plt.cm.Set3(edges_values)

    pos = nx.spring_layout(g, scale=2)

    nx.draw(g, pos)
    nx.draw_networkx_edge_labels(g, pos, edge_labels=edges_labels)
    values1 = plt.cm.Set3(edges_values)
    nx.draw_networkx_edges(g, pos, edge_color=values1, cmap=plt.get_cmap('Accent'), width=5)
#nx.draw_networkx_nodes(cc_topo, pos, label=node_labels)
    nx.draw_networkx_labels(g, pos, labels=nodes_labels)
#    nx.draw_networkx(g, node_color=set3_nodes_values, edge_color=set3_edges_values, cmap=plt.get_cmap('Accent'))
    plt.show()
