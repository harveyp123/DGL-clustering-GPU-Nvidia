import dgl
import dgl.distributed
import torch
import networkx as nx

def create_sample_graph():
    # Create a sample graph using DGL
    x0 = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8]
    # x0 = [i-1 for i in x0]
    x1 = [1, 1, 0, 2, 3, 0, 1, 3, 6, 1, 2, 4, 5, 7, 3, 5, 3, 4, 7, 2, 7, 8, 3, 5, 6, 8, 6, 7]
    # x1 = [i-1 for i in x1]
    g = dgl.graph((x0,
                   x1))
    print("original graph node information:", g.nodes())
    print("original graph edge information:", g.edges())
    return g

def partition_graph(g, num_partitions):
    # Use DGL's partition_graph function to partition the graph
    orig_nids, orig_eids = dgl.distributed.partition_graph(g, graph_name = 'test', num_parts = num_partitions, \
                                    out_path = 'test', part_method = 'metis', return_mapping= True)
    
    # Assign partition information to each node
    # g.ndata['partition'] = node_parts
    # print(g)
    # exit()

    # return g.
    # print(orig_nids)
    # exit()
    print("########## node/edge mapping: ##########")
    print("node mapping", orig_nids)
    print("shuffled node id:", g.subgraph(orig_nids).nodes())
    print("shuffled edges:", g.subgraph(orig_nids).edges())
    print("########## node/edge mapping: ##########")
    return orig_nids

def extract_partitions(orig_nids, num_partitions):
    partitions = []

    for i in range(num_partitions):
        subg, node_feat, _, _, _, _, _ = dgl.distributed.load_partition('test' + '/test.json', i)
        print("Cluster {}:".format(i),subg.ndata)
        nodes_inner = subg.filter_nodes(lambda x: x.data['inner_node'])
        nodes_outer = subg.filter_nodes(lambda x: (1 - x.data['inner_node']))

        cluster_orig_node_id = orig_nids[subg.ndata['_ID']]
        # print("cluster {} original nodes ID: ".format(i), cluster_orig_node_id)

        node_inner_orig_id = orig_nids[subg.ndata['_ID'][nodes_inner]]
        # print("inner nodes:", nodes_inner)
        # print("inner nodes original node id:", node_inner_orig_id)

        node_outer_orig_id = orig_nids[subg.ndata['_ID'][nodes_outer]]
        # print("outer nodes:", nodes_outer)
        # print("outer nodes original node id:", node_outer_orig_id)


        partitions.append({'subgraph': subg, 'subgraph_orig_node_id': cluster_orig_node_id, 'inner_nodes': nodes_inner, \
                           'outer_nodes': nodes_outer, 'inner_nodes_orig_id': node_inner_orig_id, 'outer_nodes_orig_id': node_outer_orig_id})
    return partitions

def main():
    # Create a sample graph and partition it using DGL
    g = create_sample_graph()
    num_partitions = 3
    orig_nids = partition_graph(g, num_partitions)

    # Extract partitions with inner and outer nodes and subgraphs
    partitions = extract_partitions(orig_nids, num_partitions)

    # Print information about each partition
    for i, partition in enumerate(partitions):
        print(f"Partition {i}:")
        print("cluster {} original nodes ID: ".format(i), partition['subgraph_orig_node_id'])
        print(f"  Inner nodes: {partition['inner_nodes']}")
        print(f"  Inner nodes original node id: {partition['inner_nodes_orig_id']}")
        print(f"  Outer nodes: {partition['outer_nodes']}")
        print(f"  Outer nodes original node id: {partition['outer_nodes_orig_id']}")

if __name__ == "__main__":
    main()