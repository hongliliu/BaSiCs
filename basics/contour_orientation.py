
import numpy as np
import scipy.ndimage as nd
from skimage.morphology import medial_axis
from skimage.measure import subdivide_polygon, approximate_polygon

from basics.utils import eight_conn


def shell_orientation(mask, center, diff_thresh=0.50, verbose=False):
    '''
    Reject coordinate points if their local curvature does not point
    to the center of the ellipse.
    '''

    # Label the shells portions
    labels, num = nd.label(mask, eight_conn)
    print(num)
    inlier_coords = []
    outlier_coords = []

    # Loop through each segment, highlighting which points belong to the
    # region.
    for i in range(1, num+1):
        segment = labels == i

        # Order the points
        coords = walk_through_skeleton(segment)

        # Approx w/ splines
        coords = subdivide_polygon(coords, preserve_ends=True)
        # coords = approximate_polygon(coords, tolerance=2)

        # Can't calculate with < 3 points.
        if len(coords) < 3:
            outlier_coords.extend(coords)
            continue

        y, x = coords.T

        num_diff = 5

        for j in range(num_diff, len(coords) - num_diff):

            pt1 = coords[j-num_diff]
            pt2 = coords[j]
            pt3 = coords[j+num_diff]

            # Find the center of the circle the passes through the points
            pt2_center = circle_center(pt1, pt2, pt3)

            # Now find the difference in the angles to the centers
            theta_real = np.arctan2(pt2[1] - center[1],
                                    pt2[0] - center[0])
            theta_curve = np.arctan2(pt2[1] - pt2_center[1],
                                     pt2[0] - pt2_center[0])

            # Find the difference of the angles
            diff_theta = np.abs(np.arctan2(np.sin(theta_curve-theta_real),
                                           np.cos(theta_curve-theta_real)))

            # import matplotlib.pyplot as p
            # p.imshow(segment, origin="lower")
            # p.plot(coords[:, 1], coords[:, 0], 'mo')
            # p.plot(pt1[1], pt1[0], 'bD')
            # p.plot(pt2[1], pt2[0], 'rD')
            # p.plot(pt3[1], pt3[0], 'gD')
            # p.show()

            # before/after
            if diff_theta < diff_thresh:
                if j == num_diff:
                    inlier_coords.extend(coords[:num_diff])
                elif j == len(coords) - num_diff - 1:
                    inlier_coords.extend(coords[-num_diff:])
                inlier_coords.append(pt2)
            else:
                if j == num_diff:
                    outlier_coords.extend(coords[:num_diff])
                elif j == len(coords) - num_diff - 1:
                    outlier_coords.extend(coords[-num_diff:])
                outlier_coords.append(pt2)

    # Convert to arrays
    inlier_coords = np.array(inlier_coords)
    outlier_coords = np.array(outlier_coords)

    if verbose:
        import matplotlib.pyplot as p

        p.imshow(mask, origin='lower', interpolation='nearest')
        p.plot(inlier_coords[:, 1], inlier_coords[:, 0], 'bo')
        p.plot(outlier_coords[:, 1], outlier_coords[:, 0], 'ro')
        p.plot(center[1], center[0], 'gD')

        p.show()

    return inlier_coords, outlier_coords


end_structs = [np.array([[1, 0, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 1, 0],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 1],
                         [0, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [1, 1, 0],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 1],
                         [0, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [1, 0, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 1, 0]]),
               np.array([[0, 0, 0],
                         [0, 1, 0],
                         [0, 0, 1]])]

four_conn_posns = [1, 3, 5, 7]
eight_conn_posns = [0, 2, 6, 8]


def circle_center(pt1, pt2, pt3):
    '''
    Find the center of the circle through 3 points.
    '''

    norm12 = np.linalg.norm(pt1-pt2)
    norm13 = np.linalg.norm(pt1-pt3)
    norm23 = np.linalg.norm(pt3-pt2)

    b1 = norm23**2 * (norm13**2 + norm12**2 - norm23**2)
    b2 = norm13**2 * (norm23**2 + norm12**2 - norm13**2)
    b3 = norm12**2 * (norm23**2 + norm13**2 - norm12**2)

    P = np.vstack([pt1, pt2, pt3]).T.dot(np.vstack([b1, b2, b3]))
    P /= b1 + b2 + b3

    return P


def walk_through_skeleton(skeleton):
    '''
    Starting from one end, walk through a skeleton in order. Intended for use
    with skeletons that contain no branches.
    '''

    # Calculate the end points
    end_pts = return_ends(skeleton)
    if len(end_pts) != 2:
        raise ValueError("Skeleton must contain no intersections.")

    all_pts = int(np.sum(skeleton))

    yy, xx = np.mgrid[-1:2, -1:2]
    yy = yy.ravel()
    xx = xx.ravel()

    for i in xrange(all_pts):
        if i == 0:
            ordered_pts = [end_pts[0]]
            prev_pt = end_pts[0]
        else:
            # Check for neighbors
            y, x = prev_pt
            # Extract the connected region
            neighbors = skeleton[y-1:y+2, x-1:x+2].ravel()
            # Define the corresponding array indices.
            yy_inds = yy + y
            xx_inds = xx + x

            hits = [int(elem) for elem in np.argwhere(neighbors)]
            # Remove the centre point and any points already in the list
            for pos, (y_ind, x_ind) in enumerate(zip(yy_inds, xx_inds)):
                if (y_ind, x_ind) in ordered_pts:
                    hits.remove(pos)

            num_hits = len(hits)

            if num_hits == 0:
                # You've reached the end. It better be the other end point
                if prev_pt[0] != end_pts[1][0] or prev_pt[1] != end_pts[1][1]:
                    raise ValueError("Final point does not match expected"
                                     " end point. Check input skeleton for"
                                     " intersections.")
                ordered_pts.append(end_pts[1])
                break
            elif num_hits == 1:
                # You have found the next point
                posn = hits[0]
                next_pt = (y+yy[posn], x+xx[posn])
                ordered_pts.append(next_pt)
            else:
                # There's at least a couple neighbours (for some reason)
                # Pick the 4-connected component since it is the closest
                for fours in four_conn_posns:
                    if fours in hits:
                        posn = hits[hits.index(fours)]
                        break
                else:
                    raise ValueError("Disconnected eight-connected pixels?")
                next_pt = (y+yy[posn], x+xx[posn])
                ordered_pts.append(next_pt)
            prev_pt = next_pt

    return np.array(ordered_pts)


def return_ends(skeleton):
    '''
    Find the endpoints of the skeleton.
    '''

    end_points = []

    for i, struct in enumerate(end_structs):
        hits = nd.binary_hit_or_miss(skeleton, structure1=struct)

        if not np.any(hits):
            continue

        for y, x in zip(*np.where(hits)):
            end_points.append((y, x))

    return end_points
