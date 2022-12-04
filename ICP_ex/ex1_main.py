import ex1 as ex
import numpy as np
if __name__ == '__main__':
    Data = np.load('icp_data.npz')
    Line1 = Data['LineGroundTruth']
    Line2 = Data['LineMovedNoCorresp']

    MaxIter = 100
    Epsilon = 0.2
    E = np.inf
    # show figure
    ex.show_figure(Line1, Line2)

    for i in range(MaxIter):

        # TODO: find correspondences of points
        # point with index QInd(1, k) from Line1 corresponds to
        # point with index PInd(1, k) from Line2
        PInd = np.linspace(0, np.shape(Line2)[0] - 1, np.shape(Line2)[1]).astype(int)
        distances, QInd = ex.nearest_neighbor(np.transpose(Line2), np.transpose(Line1))

        # update Line2 and error
        # Now that you know the correspondences, use your implementation
        # of icp with known correspondences and perform an update
        EOld = E
        [Line2, E] = ex.icp_known_corresp(Line1, Line2, QInd, PInd)

        print('Error value on ' + str(i) + ' iteration is: ', E)
        if E<0.3:
            ex.show_figure(Line1, Line2)
        # TODO: perform the check if we need to stop iterating
        if E < Epsilon:
            break