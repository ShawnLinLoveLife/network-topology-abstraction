#!/usr/bin/env python2
import networkx as nx
import itertools
import random
from itertools import product
import numpy as np
from networkx import connected_components
import sys
import fnss
from operator import itemgetter
from sets import Set
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
import sys
import shutil

#G = nx.read_graphml("./Chinanet.graphml")
#shortest_path_map = nx.all_pairs_shortest_path(G)
#average_shortest_path_length = nx.average_shortest_path_length(G)


def origin_average_shortest_path(G, host_list, cost_func=None):
    path_sum = 0
    path_pair = 0
    for src_dst_pair in list(itertools.permutations(host_list, 2)):
        path_sum += cost_func(G, src_dst_pair[0], src_dst_pair[1])
        path_pair += 1
    return path_sum*1.0/path_pair


def find_host_nodes_new(graph, strongly_connected_component):
    #print(strongly_connected_component)
    host_list = []
    for node in graph.nodes(data=True):
        #if node[1]['Internal'] == 1 or node[1]['Internal'] == True: # Filter the node has not enough information
            if 1==1:#nx.degree(graph, node[0]) == 1:
                #print(node)
                if strongly_connected_component:
                    if node[0] in strongly_connected_component:
                        host_list.append(node[0])
                else:
                    host_list.append(node[0])
    return host_list


def find_host_nodes(graph, strongly_connected_component):
    #print(strongly_connected_component)
    host_list = []
    for node in graph.nodes(data=True):
        if node[1]['Internal'] == 1 or node[1]['Internal'] == True: # Filter the node has not enough information
            if 1==1:#nx.degree(graph, node[0]) == 1:
                #print(node)
                if strongly_connected_component:
                    if node[0] in strongly_connected_component:
                        host_list.append(node[0])
                else:
                    host_list.append(node[0])
    return host_list


def show_nodes(G, CC_G=[]):
    for node in G.nodes(data=True):
        if node[1]['Internal'] == 1: # Filter the node has not enough information
            print(node)


def netowrk_map_average_shortest_path(G, host_list, times=1):
    # For random generated network map, we will calculate 10 times
    average_cost_between_times = 0
    for i in range(times):
        network_map = random_network_map(G, host_list)
        #print(network_map)
        cost_map = default_cost_map(G, network_map)
        #print(cost_map)
        sum_of_cost_between_pids = 0
        for v in cost_map.values():
            sum_of_cost_between_pids += v
        average_cost_between_pids = sum_of_cost_between_pids / len(cost_map)
        average_cost_between_times += average_cost_between_pids
    return average_cost_between_times / times

# Metric: average shortest path length
# Network map generator algorithm: Every node is randomly distributed into a PID
# Cost between two PID is defined as average shortest path length between a node in PID1 and a node in PID2
# PID number is: 3
# Test times is: 10

PID_NUMBER = 17


def gen_random_network_map(G, host_list, pid_number):
    network_map = []
    for i in range(pid_number):
        network_map.append([])
    for host in host_list:
        network_map[random.randint(0, pid_number-1)].append(host)
    return network_map


def gen_random_based_anchor_based_network_map(graph, CC_G, host_list, pid_number):
    network_map = []
    for i in range(pid_number):
        network_map.append([])
    # Find anchors have the largest degree
    #print(CC_G)
    anchors = []
    count = 0
    nodes = (sorted(graph.degree_iter(), key=itemgetter(1), reverse=True))
    #print(nodes)
    while count < pid_number:
        n = random.randint(0, len(nodes)-1)
        #print(n)
        #print(nodes[n])
        if nodes[n][0] in CC_G and nodes[n] not in host_list:
            anchors.append(nodes[n][0])
            count += 1
        del nodes[n]
    #print(anchors)
    for host in host_list:
        min_length = sys.maxint
        host_in_pid = -1
        for anchor in anchors:
            if nx.shortest_path_length(graph, host, anchor) < min_length:
                min_length = nx.shortest_path_length(graph, host, anchor)
                host_in_pid = anchors.index(anchor)
        if host_in_pid != -1:
            network_map[host_in_pid].append(host)
    return (anchors, network_map)


def gen_degree_based_anchor_based_network_map(graph, CC_G, host_list, pid_number):
    network_map = []
    for i in range(pid_number):
        network_map.append([])
    # Find anchors have the largest degree
    #print(CC_G)
    anchors = []
    count = 0
    nodes = (sorted(graph.degree_iter(), key=itemgetter(1), reverse=True))
    #print(nodes)
    for node in nodes:
        if node[0] in CC_G:
            anchors.append(node[0])
            count += 1
            if count == pid_number:
                break
    print(anchors)
    for host in host_list:
        min_length = sys.maxint
        host_in_pid = -1
        for anchor in anchors:
            try:
                if nx.shortest_path_length(graph, host, anchor) < min_length:
                    min_length = nx.shortest_path_length(graph, host, anchor)
                    host_in_pid = anchors.index(anchor)
            except Exception:
                continue
        if host_in_pid != -1:
            network_map[host_in_pid].append(host)
    return (anchors, network_map)


def gen_largest_weight_node_networkmap(graph, CC_G, host_list):
    network_map = []
    # Find anchors have the largest degree
    #print(CC_G)
    anchors = []
    count = 0
    nodes = (sorted(graph.degree_iter(), key=itemgetter(1), reverse=True))
    #print(nodes)

# Agis #######################################################################
#    for (u, v) in graph.edges():
#        if str(graph.edge[u][v]['label']).count('45') >= 1:
#            graph.edge[u][v]['capacity'] = 45
#        elif str(graph.edge[u][v]['label']).count('155') >= 1:
#            graph.edge[u][v]['capacity'] = 155
#        else:
#            graph.edge[u][v]['capacity'] = str('null')

#    for (u, v) in graph.edges():
#        if graph.edge[u][v]['capacity'] == 155:
#            if u not in anchors:
#                anchors.append(u)
#            if v not in anchors:
#                anchors.append(v)
##############################################################################

# Aarnet ########################################################################
#    for (u, v) in graph.edges():
#        if 'label' not in graph.edge[u][v].keys():
#            graph.edge[u][v]['capability'] = 1000
#            continue
#
#        bw = str((graph.edge[u][v]['label']))

#        unit = 1
#        if bw.count('Gbps') >= 1:
#            unit = 10 ** 9
#        elif bw.count('Mbps') >= 1:
#            unit = 10 ** 6

#        if bw.count('10'):
#            graph.edge[u][v]['capability'] = 10 * unit
#        elif bw.count('2.5'):
#            graph.edge[u][v]['capability'] = 2.5 * unit
#        elif bw.count('155'):
#            graph.edge[u][v]['capability'] = 155 * unit
#        else:
#            graph.edge[u][v]['capability'] = 1000

#    for (u, v) in graph.edges():
#        if graph.edge[u][v]['capability'] == 10**10:
#            if u not in anchors:
#                anchors.append(u)
#            if v not in anchors:
#                anchors.append(v)

# Cernet
#############################################################################
    for (u, v) in graph.edges():
        if 'label' not in graph.edge[u][v].values()[0].keys():
            continue
        bw = str((graph.edge[u][v].values()[0]['label']))
        if bw == '2.5G': # Cernet
            if u not in anchors:
                anchors.append(u)
            if v not in anchors:
                anchors.append(v)
#############################################################################
    print(anchors)
    for i in range(len(anchors)):
        network_map.append([])
    for host in host_list:
        min_length = sys.maxint
        host_in_pid = -1
        for anchor in anchors:
            try:
                if nx.shortest_path_length(graph, host, anchor) < min_length:
                    min_length = nx.shortest_path_length(graph, host, anchor)
                    host_in_pid = anchors.index(anchor)
            except Exception:
                continue
        if host_in_pid != -1:
            network_map[host_in_pid].append(host)
    return (anchors, network_map)


def default_cost_map(G, network_map):
    cost_map = {}
    for pid_pair in list(itertools.permutations(network_map, 2)):
        #print(pid_pair)
        #print(default_cost_between_pids(G, pid_pair[0], pid_pair[1]))
        #print(network_map.index(pid_pair[0]))
        #print(network_map.index(pid_pair[1]))
        cost_map.setdefault(tuple([network_map.index(pid_pair[0]),network_map.index(pid_pair[1])]), default_cost_between_pids(G, pid_pair[0], pid_pair[1]))
    return cost_map


def default_cost_between_pids(G, pid1, pid2):
    # Calculate the average shortest path between hosts in 2 pids
    result_list = []
    for i, j in product(pid1, pid2):
        result_list.append(nx.shortest_path_length(G, i, j))
    if result_list == []:
        return sys.maxint
    return np.mean(result_list)

#host = []
#pid[1] = [] nodes in PID 1
#for node in G.nodes(data=True):
#    if node[1]['Internal'] == 1: # Filter the node has not enough information
#        print(node)
#        if nx.degree(G, node[0]) == 1:
#            host.append(node[0])
        #pid[random.randint(0, PID_NUMBER-1)].append(node[0])
        #print(nx.shortest_path(G, node[0], node[0]))

#print(host)
#nx.draw(graph)
#plt.savefig("./example.png")
#plt.show()


def dist_cost_hop_count(G, cost_map, src, dst):
    # cost between host1 and host2 from cost map
    print(tuple([src,dst]))
    return 0
    return abs(cost_map.get(tuple([src,dst]))-nx.shortest_path_length(G, src, dst))


def average_dist(G, network_cost_map, host_list):
    sum_dist = 0
    network_map = network_cost_map[0]
    cost_map = network_cost_map[1]



def gen_nm_cm(G, host_list):
    network_map = gen_random_network_map(G, host_list)
    cost_map = default_cost_map(G, network_map)
    return (network_map, cost_map)


def gen_spl_mat_from_graph(graph, host_list):
    host_count = len(host_list)
    result = [[0 for i in range(host_count)] for j in range(host_count)]
    for i in range(host_count):
        for j in range(host_count):
            try:
                result[i][j] = nx.shortest_path_length(graph, host_list[i], host_list[j])
            except nx.NetworkXNoPath:
                result[i][j] = sys.maxint

    return np.asmatrix(result)


def gen_spl_mat_from_maps(network_map, cost_map, host_list):
    host_count = len(host_list)
    result = [[0 for i in range(host_count)] for j in range(host_count)]
    for i in range(host_count):
        for j in range(host_count):
            if i == j:
                result[i][j] = 0
            else:
                i_in_pid = 0
                j_in_pid = 0
                is_found = 0
                for pid in network_map:
                    if host_list[i] in pid:
                        i_in_pid = network_map.index(pid)
                        is_found += 1
                    if host_list[j] in pid:
                        j_in_pid = network_map.index(pid)
                        is_found += 1
                    if is_found == 2:
                        break
                if is_found != 2:
                    result[i][j] = sys.maxint
                if i_in_pid == j_in_pid:
                    result[i][j] = 0
                else:
                    result[i][j] = cost_map.get((i_in_pid, j_in_pid))
    return np.asmatrix(result)


def show_edge(G):
    for edge in G.edges:
        print(edge)


def select_largest_cc(topology):
    largest = reduce(lambda x, y: x if len(x) > len(y) else y, connected_components(topology))
    to_be_removed = set(topology.nodes()) - set(largest)
    topology.remove_nodes_from(to_be_removed)

    return topology


def gen_girvan_newman():
################################################
# "./Chinanet.graphml"
    print("begin")
# path = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/Kdl.graphml"
    path_root = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/"
# path = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/Kdl.graphml"
    topology_name = "Cernet"
    path = path_root + topology_name + ".graphml"
# pretreatment undirected graph, connected_components
    G_UN = nx.read_graphml(path).to_undirected()
    CC_G = max(nx.connected_components(G_UN), key=len)

    host_list = find_host_nodes(G_UN, CC_G)
################################################
    topology_CC_G = select_largest_cc(G_UN)
    #nx.draw_networkx(topology_CC_G)
    #plt.show()
    print(nx.edge_betweenness_centrality(topology_CC_G))
    central = sorted(nx.edge_betweenness_centrality(topology_CC_G).items(), key=lambda d: d[1], reverse=True)
    print sorted(nx.edge_betweenness_centrality(topology_CC_G).items(), key=lambda d: d[1], reverse=True)

    print(central)
    #print(central)
    #print(nx.edge_betweenness_centrality(topology_CC_G))g
    while len(central) != 0 and len(sorted(nx.connected_components(topology_CC_G), key = len, reverse=True)) < 10:
        topology_CC_G.remove_edge(central[0][0][0], central[0][0][1])
        #print(sorted(nx.connected_components(topology_CC_G), key = len, reverse=True))
#        print(topology_CC_G)

        central = sorted(nx.edge_betweenness_centrality(topology_CC_G).items(), key=lambda d: d[1], reverse=True)
        print(central)
    nx.draw_networkx(topology_CC_G)
    plt.show()


def get_link_bandwidth(topology, edge):
    bandwidth1 = str(topology[edge[0]][edge[1]].values()[0].get('label'))
    bandwidth2 = str(topology[edge[1]][edge[0]].values()[0].get('label'))
    if bandwidth1 == 'External' or bandwidth2 == 'External' or bandwidth1 == 'None' or bandwidth2 == 'None':
        return 0
    else:
        return min(float_bandwidth(bandwidth1), float_bandwidth(bandwidth2))


def float_bandwidth(bandwidth):
    if bandwidth.endswith("G"):
        return float(bandwidth[0:-1])*10**9
    elif bandwidth.endswith("M"):
        return float(bandwidth[0:-1])*10**6
    else:
        return float(bandwidth[0:-1])*10**3


def girvan_newman_example():
    print("begin")
    path_root = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/"
    topology_name = "Cernet"
    path = path_root + topology_name + ".graphml"
    G_UN = nx.read_graphml(path).to_undirected()
    CC_G = max(nx.connected_components(G_UN), key=len)
    host_list = find_host_nodes(G_UN, CC_G)
    topology_CC_G = select_largest_cc(G_UN)
    topology_CC_G_1 = topology_CC_G
    a = topology_CC_G.edges()
#    for (u, v) in topology_CC_G.edges():
#        print(topology_CC_G[u][v]['label'])
#        bw = topology_CC_G[u][v].values()[0].get('label')
        #topology_CC_G[u][v]['bandwidth'] = bw


    import itertools
    k = 4
    comp = nx.girvan_newman(topology_CC_G)
    result = ()
    limited = itertools.takewhile(lambda c: len(c) <= k, comp)
    for communities in limited:
        result = tuple(sorted(c) for c in communities)
    print(result)


    ###Print result
    values = []
    for node in topology_CC_G.nodes(data=True):
        for pid in result:
            if node[0] in pid:
                print(node[0])
                values.append((result.index(pid)+1)/10.0)
    values1 = plt.cm.Set3(values)
    print(values)
    #plt.figure(figsize=(20, 20))
    nx.draw_networkx(topology_CC_G, node_color=values1, cmap=plt.get_cmap('Accent'))
    #plt.show()


def print_graph(G, anchors, network_map, topology_name, method_name, times=0, dist=0):
    # filename = topology name _ anchors number _ method
    values = []
    node_size = []
    node_count = 0
    for node in G.nodes(data=True):
        node_count = node_count+1
        #print(node)
        if node[0] in anchors:
            values.append((anchors.index(node[0])+1)/10.0)
            node_size.append(800)
        else:
            flag = True
            for pid in network_map:
                if node[0] in pid:
                    values.append((network_map.index(pid)+1)/10.0)
                    node_size.append(400)
                    flag = False
                    break
            if flag:
                values.append(1)
                node_size.append(1)
    #print(values)
    pid_count = len(network_map)
    compression_ratio = pid_count*1.0/node_count
    values1 = plt.cm.Set3(values)
    #print(values1)
    #print(np.linspace(0, 1, 12))
    #print(plt.cm.Set3(np.linspace(0, 1, 12)))
    plt.figure(figsize=(20, 20))
    nx.draw_networkx(G, node_color=values1, cmap=plt.get_cmap('Accent'), with_labels=False, node_size=node_size)
    picture_name = "/" + topology_name + '_' + str(len(anchors)) + '_' + method_name + '_' + str(times) + '_' + str(dist) + '_' + str(compression_ratio) + ".png"
    #if os.path.exists("./result/"+topology_name):
    #    os.remove("./result/"+topology_name)
    #os.mkdir("./result/"+topology_name)
    plt.savefig("./result/" + topology_name + picture_name)
    #plt.savefig("graph.png", dpi=1000)
    #plt.show()


def main():
    # "./Chinanet.graphml"
    print("begin")
    #path = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/Kdl.graphml"
    path_root = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/"
    #path = "/Users/shawn/Develop/networkmap-evaluation/rsa-eval/dataset/topologyzoo/sources/Kdl.graphml"
    topology_name = "Cernet"
    path = path_root + topology_name + ".graphml"
    # pretreatment undirected graph, connected_components
    G_UN = nx.read_graphml(path).to_undirected()
    CC_G = max(nx.connected_components(G_UN), key=len)

    host_list = find_host_nodes(G_UN, CC_G)
    origin_spl_mat = gen_spl_mat_from_graph(G_UN, host_list)
    anchor_number = 3
    if os.path.exists('./result/'+topology_name):
        #os.remove('./result/'+topology_name)
        shutil.rmtree('./result/'+topology_name, ignore_errors=False, onerror=None)
    os.mkdir('./result/'+topology_name)
    if os.path.exists('./result/'+topology_name+'/' + topology_name + "_" + str(anchor_number)+".txt"):
        os.remove('./result/'+topology_name+'/' + topology_name + "_" + str(anchor_number)+".txt")
    f = open('./result/'+topology_name+'/' + topology_name + "_" + str(anchor_number)+".txt", 'a+')

    #network_map = gen_random_network_map(G_UN, host_list, 3)
    #cost_map = default_cost_map(G_UN, network_map=network_map)
    #print(network_map)
    #print(network_map)
    #print(cost_map)
    #random_spl_mat = gen_spl_mat_from_maps(network_map, cost_map, host_list)
    #print(origin_spl_mat)
    #print(random_spl_mat)
    #dist = (np.absolute(origin_spl_mat-random_spl_mat)).sum()
    #print(dist)

    print >> f, "============================degree=========================================="
    # Anchors based network map
    print("degreee based anchor based network map")
    anchor_based_network_map = gen_degree_based_anchor_based_network_map(G_UN, CC_G, host_list, 17)
    anchor_based_cost_map = default_cost_map(G_UN, network_map=anchor_based_network_map[1])
    anchor_spl_mat = gen_spl_mat_from_maps(anchor_based_network_map[1], anchor_based_cost_map, host_list)
    #print(anchor_spl_mat)
    print >> f, anchor_based_network_map[0]
    print >> f, anchor_based_network_map[1]
    dist = (np.absolute(origin_spl_mat-anchor_spl_mat)).sum()/len(host_list)/len(host_list)
    # anchor_spl_mat.sum()/origin_spl_mat.sum()
    print_graph(G_UN, anchor_based_network_map[0], anchor_based_network_map[1], topology_name, "degree_based", times=0, dist=dist)
    print >> f, dist
    random_dist_list = []
    TIMES = 100 # Random select anchor TIMES times
    random_100 = range(TIMES)
    random_print_list = []
    for i in range(10):
        selected = random.choice(random_100)
        random_print_list.append(selected)
        random_100.remove(selected)
    print(random_print_list)
    print >> f, random_print_list
    for i in range(TIMES):
        print(i)
        print >> f, ("==============================random===========================================")
        random_anchor_based_network_map = gen_random_based_anchor_based_network_map(G_UN, CC_G, host_list, 17)
        random_anchor_based_cost_map = default_cost_map(G_UN, network_map=random_anchor_based_network_map[1])
        random_anchor_spl_mat = gen_spl_mat_from_maps(random_anchor_based_network_map[1], random_anchor_based_cost_map, host_list)
        print >>f, (random_anchor_based_network_map[0])
        print >>f, (random_anchor_based_network_map[1])
        dist = (np.absolute(origin_spl_mat-random_anchor_spl_mat)).sum()/len(host_list)/len(host_list)
        #random_anchor_spl_mat.sum()/origin_spl_mat.sum()

        random_dist_list.append(dist)
        print >>f, (dist)
        if i in random_print_list:
            print_graph(G_UN, random_anchor_based_network_map[0], random_anchor_based_network_map[1], topology_name, "random_based", random_print_list.index(i), dist)
    print >> f, "=================average dist for random=============================================="
    print >> f, np.mean(random_dist_list)

    print >> f, "=======================================largest weight================================================"
    print("largest weight based")
    largest_weight_network_map = gen_largest_weight_node_networkmap(G_UN, CC_G, host_list)
    largest_weight_cost_map = default_cost_map(G_UN, network_map=largest_weight_network_map[1])
    largest_weight_spl_mat = gen_spl_mat_from_maps(largest_weight_network_map[1], largest_weight_cost_map, host_list)
    print >> f, largest_weight_network_map[0]
    print >> f, largest_weight_network_map[1]
    dist = (np.absolute(origin_spl_mat-largest_weight_spl_mat)).sum()/len(host_list)/len(host_list)
    print_graph(G_UN, largest_weight_network_map[0], largest_weight_network_map[1], topology_name, "largest_weight", times=0,
                dist=dist)
    #CC_G = max(nx.connected_components(G), key=len)
    #G = nx.read_graphml(path)

    #host_list = find_host_nodes(G, CC_G)
    #average_shortest_path_length = origin_average_shortest_path(G, host_list, cost_func=lambda G, x, y: nx.shortest_path_length(G, x, y))
    #print("original average shortest path length: %f" % average_shortest_path_length)
    #network_map_average_shortest_path_length = netowrk_map_average_shortest_path(G, host_list, 10)
    #print("random select: %f" % network_map_average_shortest_path_length)
    #print(gen_nm_cm(G, host_list)[1])
    #print("average dist: %f" % average_dist(G, average_dist(G, network_cost_map=gen_nm_cm(G, host_list))))

main()

#girvan_newman_example()
