import ex2 as ex
def Task0():
    import ex2 as ex
    # load a dataset
    filename = 'data/simulation-pose-landmark.g2o'
    graph = ex.read_graph_g2o(filename)

    # visualize the dataset
    ex.plot_graph(graph)
    print('Loaded graph with {} nodes and {} edges'.format(len(graph.nodes), len(graph.edges)))

    # print information for the two types of nodes
    nodeId = 128
    print('Node {} = {} is a VERTEX_SE2 node'.format(nodeId, graph.nodes[nodeId]))

    # access the state vector using the lookup table
    fromIdx = graph.lut[nodeId]
    print('Node {} from the state vector = {}'.format(nodeId, graph.x[fromIdx:fromIdx + 3]))

    nodeId = 1
    print('Node {} = {} is a VERTEX_XY node'.format(nodeId, graph.nodes[nodeId]))

    # access the state vector using the lookup table
    fromIdx = graph.lut[nodeId]
    print('Node {} from the state vector = {}'.format(nodeId, graph.x[fromIdx:fromIdx + 2]))

    # print information for two types of edges
    eid = 0
    print('Edge {} = {} is a pose-pose constraint'.format(eid, graph.edges[eid]))

    eid = 1
    print('Edge {} = {} is a pose-landmark constraint'.format(eid, graph.edges[eid]))
def Task1():
    import ex2 as ex
    # load a dataset
    filename = 'data/simulation-pose-landmark.g2o'
    graph = ex.read_graph_g2o(filename)
    ex.compute_global_error(graph)
def Task2():
    filename = 'data/simulation-pose-pose.g2o'
    g = ex.read_graph_g2o(filename)
    for edge in g.edges:

        # pose-pose constraint
        if edge.Type == 'P':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node state for the current edge
            x_i = g.x[fromIdx:fromIdx + 3]
            x_j = g.x[toIdx:toIdx + 3]

            e, A, B = ex.linearize_pose_pose_constraint(x_i, x_j, edge.measurement)
            print(B)
def Task3():
    filename = 'data/simulation-pose-landmark.g2o'
    g = ex.read_graph_g2o(filename)
    for edge in g.edges:

        if edge.Type == 'L':

            # compute idx for nodes using lookup table
            fromIdx = g.lut[edge.fromNode]
            toIdx = g.lut[edge.toNode]

            # get node states for the current edge
            x = g.x[fromIdx:fromIdx + 3]
            l = g.x[toIdx:toIdx + 2]

            e, A, B = ex.linearize_pose_landmark_constraint(
                x, l, edge.measurement)
def Task4():
    filename = 'data/dlr.g2o'
    g = ex.read_graph_g2o(filename)
    ex.run_graph_slam(g,100)
if __name__ == '__main__':
    Task4()