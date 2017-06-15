#!/usr/bin/env python2

import networkx as nx
from networkx import connected_components
import matplotlib.pyplot as plt
from operator import itemgetter
import math
import time
import itertools
import statistics

def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology


def show_graph_basic_info(old_topo, new_topo, eu_dist, exec_time):
    print("=========Basic Info========")
    print(new_topo.name)
    print("node_number %d" % new_topo.number_of_nodes())
    print("edge_number %d" % new_topo.number_of_edges())
    print("node_compression_ratio %.2f%%" % (100.0*new_topo.number_of_nodes()/old_topo.number_of_nodes()))
    print("edge_compression_ratio %.2f%%" % (100.0*new_topo.number_of_edges()/old_topo.number_of_edges()))
    print("euclidean_distance %f" % eu_dist)
    print("exec time %f" % exec_time)

def show_nodes(cc_topo):
    print("========Node========")
    for u in cc_topo.nodes():
        print(u)
        print(cc_topo[u])


def dist_from_original_graph(cc_topo, u, v):
    return nx.shortest_path_length(cc_topo, u, v, weight='latency')


def find_u_in_new_graph(spt_topo, node_mapping_set, u):
    if u not in spt_topo.nodes():
        for key, value in node_mapping_set.items():
            if u in value:
                return key
    return u


def dist_from_aggregated_graph(spt_topo, node_mapping_set, u, v):
    u = find_u_in_new_graph(spt_topo, node_mapping_set, u)
    v = find_u_in_new_graph(spt_topo, node_mapping_set, v)
    return nx.shortest_path_length(spt_topo, u, v, weight='latency')


def euclidean_distance_between_graph(old_graph, new_graph, node_mapping_set):
    result = 0
    for u in old_graph.nodes():
        for v in old_graph.nodes():
            result = result + (dist_from_original_graph(old_graph, u, v)-dist_from_aggregated_graph(new_graph, node_mapping_set, u, v))**2
    return math.sqrt(result)


def show_edges(cc_topo):
    print("========Link========")
    for (u, v) in cc_topo.edges():
        print(u, v)
        print(cc_topo[u][v])


def cernet_ensure_latency(cc_topo):
    for (u, v) in cc_topo.edges():
        if 'label' not in cc_topo.edge[u][v].keys():
            cc_topo.edge[u][v].setdefault('label', '2M')
        bw = str(cc_topo.edge[u][v]['label'])
        if bw == '2M' or bw == 'External':
            cc_topo.edge[u][v]['latency'] = 2
        elif bw == '155M':
            cc_topo.edge[u][v]['latency'] = 10
        elif bw == '2.5G':
            cc_topo.edge[u][v]['latency'] = 20


def draw_topology(cc_topo, type):
    links_values = []
    node_labels = {}
    edge_labels = {}
    for (u, v) in cc_topo.edges():
        if type in cc_topo.edge[u][v].keys():
            type_value = cc_topo.edge[u][v][type]
            if type == 'latency':
                edge_labels.setdefault((u, v), type_value)
                test_list = []
                test_list.append(abs(type_value-2))
                test_list.append(abs(type_value-10))
                test_list.append(abs(type_value-20))
                index = test_list.index(min(test_list))
                type_value = 2 if index == 0 else 10 if index == 1 else 20
                if type_value == 2:
                    links_values.append(1 / 3.0)
                elif type_value == 10:
                    links_values.append(2 / 3.0)
                elif type_value == 20:
                    links_values.append(3 / 3.0)
    for u in cc_topo.nodes():
        node_labels.setdefault(u, u)
    pos = nx.spring_layout(cc_topo, scale=2)
    values1 = plt.cm.Set3(links_values)
    plt.figure(figsize=(20, 20))
    nx.draw(cc_topo, pos)
    nx.draw_networkx_edges(cc_topo, pos, edge_color=values1, cmap=plt.get_cmap('Accent'), width=5)
    nx.draw_networkx_edge_labels(cc_topo, pos, edge_labels=edge_labels)
    nx.draw_networkx_labels(cc_topo, pos, labels=node_labels)
    plt.savefig("./result/" + cc_topo.name)
    plt.clf()


def find_max_degree_centrality(cc_topo):
    nodes = (sorted(cc_topo.degree_iter(), key=itemgetter(1), reverse=True))
    return nodes[0][0]


def find_max_closeness_centrality(cc_topo):
    nodes = nx.closeness_centrality(cc_topo)
    return sorted(nodes.items(), key=lambda d:d[1], reverse=True)[0][0]


def find_max_betweenness_centrality(cc_topo):
    nodes = nx.betweenness_centrality(cc_topo)
    return sorted(nodes.items(), key=lambda d: d[1], reverse=True)[0][0]


def gen_shortest_path_tree_topo(root, cc_topo):
    edges = []
    nodes = []
    print(root)
    dist_u_map = nx.shortest_path_length(cc_topo, source=root, weight='latency')
    #print(dist_u_map)
    for node_u in cc_topo.nodes():
        if node_u != root and node_u not in nodes:
            pu_map = {} # key is node, and value is the distance between node to root
            for node_pu in cc_topo.nodes():
                if cc_topo.has_edge(node_pu, node_u):
                    dist_pu = dist_u_map.get(node_pu)
                    edge_dist_pu_u = nx.shortest_path_length(cc_topo, node_pu, node_u, weight='latency')
                    dist_u = dist_u_map.get(node_u)
                    if dist_pu + edge_dist_pu_u == dist_u:
                        pu_map.setdefault(node_pu, nx.shortest_path_length(cc_topo, node_pu, root, weight='latency'))
            #print(pu_map)
            if len(pu_map) > 0:
                pu = sorted(pu_map.items(), key=lambda d: d[1])[0][0]
                edges.append((node_u, pu))
    #print(edges)
    spt_topo = nx.Graph(cc_topo)
    for (u, v) in spt_topo.edges():
        if (u, v) not in edges and (v, u) not in edges:
            spt_topo.remove_edge(u, v)
    spt_topo.name = "spt_" + cc_topo.name
    return spt_topo


def check_latency(cc_topo, latency):
    for (u, v) in cc_topo.edges():
        if cc_topo[u][v]['latency'] < latency:
            return False
    return True


def aggregate_nodes_by_latency(cc_topo, root, latency):
    aggregated_spt_topo = nx.Graph(cc_topo)
    # record node relationship
    node_mapping_set = {}
    for node in aggregated_spt_topo.nodes():
        node_mapping_set.setdefault(node, set([node]))

    while not check_latency(aggregated_spt_topo, latency):
        show_edges(aggregated_spt_topo)
        for (u, v) in aggregated_spt_topo.edges():
            if u not in aggregated_spt_topo.nodes() or v not in aggregated_spt_topo.nodes():
                continue
            if not aggregated_spt_topo.has_edge(u, v) and not aggregated_spt_topo.has_edge(v, u):
                continue
            if aggregated_spt_topo.edge[u][v]['latency'] < latency:
                dist_u_root = nx.shortest_path_length(aggregated_spt_topo, u, root, weight='latency')
                dist_v_root = nx.shortest_path_length(aggregated_spt_topo, v, root, weight='latency')
                if dist_u_root > dist_v_root:
                    t = u
                    u = v
                    v = t
                temp_set = node_mapping_set.get(u) | node_mapping_set.get(v)
                node_mapping_set.pop(v)
                node_mapping_set.pop(u)
                node_mapping_set.setdefault(u, temp_set)
                v_neighbors = aggregated_spt_topo.neighbors(v)
                #print("==v")
                #print(v)
                #print("==neighbors")
                #print(v_neighbors)
                for v_neighbor in v_neighbors:
                    if u != v_neighbor:
                        if not aggregated_spt_topo.has_edge(u, v_neighbor) and not aggregated_spt_topo.has_edge(v_neighbor, u):
                            aggregated_spt_topo.add_edge(u, v_neighbor, aggregated_spt_topo.edge[v][v_neighbor])
                        aggregated_spt_topo.remove_edge(v, v_neighbor)
                        if aggregated_spt_topo.has_edge(v_neighbor, v):
                            aggregated_spt_topo.remove_edge(v_neighbor, v)
                aggregated_spt_topo.remove_node(v)
                if len(aggregated_spt_topo.neighbors(u)) == 0:
                    aggregated_spt_topo.remove_node(u)
        #show_edges(aggregated_spt_topo)
    aggregated_spt_topo.name = "aggregated_" + cc_topo.name
    #show_nodes(aggregated_spt_topo)
    #show_edges(aggregated_spt_topo)
    ### test set
    #print(len(reduce(lambda x, y: x | y, node_mapping_set.values())))
    return aggregated_spt_topo, node_mapping_set


def add_original_edge_back(old_topo, aggregated_spt_topo, node_mapping_set, root):
    print("======================add back link")
    add_edge_back_aggregated_topo = nx.Graph(aggregated_spt_topo)
    add_edge_back_aggregated_topo.name="add_edge_back_" + aggregated_spt_topo.name
    for u in add_edge_back_aggregated_topo.nodes():
        for v in add_edge_back_aggregated_topo.nodes():
            if not add_edge_back_aggregated_topo.has_edge(u, v) and not add_edge_back_aggregated_topo.has_edge(v, u) and u != v:
                u_set = node_mapping_set.get(u)
                v_set = node_mapping_set.get(v)
                cost_list = []
                for ue in u_set:
                    for ve in v_set:
                        cost_list.append(nx.shortest_path_length(old_topo, ue, ve, weight='latency'))
                new_edge_weight = statistics.median(cost_list)
                #new_edge_weight = statistics.mean(cost_list)
                if new_edge_weight > abs(nx.shortest_path_length(add_edge_back_aggregated_topo, u, root, weight='latency') - nx.shortest_path_length(add_edge_back_aggregated_topo, v, root, weight='latency')):
                    #print((u, v))
                    add_edge_back_aggregated_topo.add_edge(u, v, {'latency': new_edge_weight})
    return add_edge_back_aggregated_topo


topology_name = "Cernet"
filename = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/" + topology_name + ".graphml"
topo = nx.read_graphml(filename).to_undirected()
cc_topo_di = select_largest_cc(topo)
cc_topo = nx.Graph(cc_topo_di)
cc_topo.name = topology_name

show_nodes(topo)
show_nodes(cc_topo)

print(find_max_degree_centrality(cc_topo))
print(find_max_closeness_centrality(cc_topo))
print(find_max_betweenness_centrality(cc_topo))

#print_graph_basic_info(cc_topo)

cernet_ensure_latency(cc_topo)


root = find_max_degree_centrality(cc_topo)
start = time.clock()
spt_topo = gen_shortest_path_tree_topo(root, cc_topo)
end = time.clock()
print("=================gen_shortest_path_tree_time")
gen_spt_time = end - start
print(gen_spt_time)


start = time.clock()
aggregated_spt_topo = aggregate_nodes_by_latency(spt_topo, root, 11)
end = time.clock()
print("=================aggregate_gen_shortest_path_tree")
aggregate_spt_time = end - start
print(aggregate_spt_time)

start = time.clock()
add_edge_back_aggregated_spt_topo = add_original_edge_back(cc_topo, aggregated_spt_topo[0], aggregated_spt_topo[1], root)
end = time.clock()
print("=================add_edge_back")
add_edge_back_time = end - start
print(add_edge_back_time)


node_mapping_set = aggregated_spt_topo[1]
print("=======================node mapping set")
print(node_mapping_set)

show_graph_basic_info(cc_topo, spt_topo, euclidean_distance_between_graph(cc_topo, spt_topo, aggregated_spt_topo[1]), gen_spt_time)
show_graph_basic_info(cc_topo, aggregated_spt_topo[0], euclidean_distance_between_graph(cc_topo, aggregated_spt_topo[0], aggregated_spt_topo[1]), gen_spt_time+aggregate_spt_time)
show_graph_basic_info(cc_topo, add_edge_back_aggregated_spt_topo, euclidean_distance_between_graph(cc_topo, add_edge_back_aggregated_spt_topo, aggregated_spt_topo[1]), gen_spt_time+aggregate_spt_time+add_edge_back_time)
#show_graph_basic_info(cc_topo, aggregated_spt_topo[0], euclidean_distance_between_graph(cc_topo, aggregated_spt_topo[0], aggregated_spt_topo[1]), gen_spt_time+aggregate_spt_time)


draw_topology(cc_topo, 'latency')
draw_topology(spt_topo, 'latency')
draw_topology(aggregated_spt_topo[0], 'latency')
draw_topology(add_edge_back_aggregated_spt_topo, 'latency')

#print("========================origin spt")
#print(euclidean_distance_between_graph(cc_topo, spt_topo, aggregated_spt_topo[1]))
#print("========================origin aggregated spt")
#print(euclidean_distance_between_graph(cc_topo, aggregated_spt_topo[0], aggregated_spt_topo[1]))
#print("========================add link back origin aggregated spt")
#print(euclidean_distance_between_graph(cc_topo, add_edge_back_aggregated_spt_topo, aggregated_spt_topo[1]))



# for (u, v) in cc_topo.edges():
#     flag = 0
#     if 'latency' in cc_topo.edge[u][v].values()[0].keys():
#         flag = 1
#     if 'label' not in (cc_topo.edge[u][v].values()[flag]).keys():
#         links_values.append(4/4.0)
#         edge_labels.setdefault((u, v), 'null')
#         continue
#     bw = str((cc_topo.edge[u][v].values()[0]['label']))
#     edge_labels.setdefault((u, v), bw)
#     if bw == '2M':
#         links_values.append(1/4.0)
#         cc_topo.edge[u][v]['latency'] = 2
#     elif bw == '155M':
#         links_values.append(2/4.0)
#         cc_topo.edge[u][v]['latency'] = 10
#     elif bw == '2.5G':
#         links_values.append(3/4.0)
#         cc_topo.edge[u][v]['latency'] = 20
#     else:
#         links_values.append(4/4.0)
#         cc_topo.edge[u][v]['latency'] = 2

# for u in cc_topo.nodes():
#     node_labels.setdefault(u, u)
#
#
# print("========Link========")
# for (u, v) in cc_topo.edges():
#     print(cc_topo[u][v])
#     print(cc_topo[u][v].values())

# print(node_labels)
# print(links_values)
# pos = nx.spring_layout(cc_topo, scale=2)
# print(pos)
# nx.draw(cc_topo, pos)
# nx.draw_networkx_edge_labels(cc_topo, pos, edge_labels=edge_labels)
# values1 = plt.cm.Set3(links_values)
# #nx.draw_networkx_edges(cc_topo, pos, edge_color=values1, cmap=plt.get_cmap('Accent'), width=5)
# #nx.draw_networkx_nodes(cc_topo, pos, label=node_labels)
# nx.draw_networkx_labels(cc_topo, pos, labels=node_labels)
# print(edge_labels)
# #nx.draw_networkx(cc_topo, edge_color=values1, cmap=plt.get_cmap('Accent'), edge_labels=edge_labels)
# #plt.show()

