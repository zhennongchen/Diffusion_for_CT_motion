# because of the in-plane rotation (rz), we have angle overlapping problem
# solution:
# Calculate the weights for each projection
# first find the direction of each projection by applying the rotation matrix (in this case, it is just the rotation angle)
# then align all the angles sequentially
# at last, the weight (d_theta) will the average between the distance to the angle in front and the next one.

# work for (1) in-plane motion, (2) parallel beam, (3) any SGA
# not work for (1) through-plane motion (small residual artifacts) （2） fan-beam (large artifacts)

import numpy as np

def weighted_projection_due_to_rotation(sinogram_segments, segment_indexes, spline_rz, total_views, num_of_pars,times, angles, sga, use_affine_transform = True):

    prj_dirs = []
    nview_per_step = total_views // num_of_pars

    for istep in range(0,num_of_pars):
        tt = times[istep]
        rot_angle = spline_rz([tt])
    
        current_angles = angles[istep * nview_per_step:(istep + 1) * nview_per_step] - sga / 180 * np.pi
        if use_affine_transform == True:
            new_angles = current_angles - rot_angle  
        else:
             new_angles = current_angles + rot_angle   # use scipy.ndimage.rotate

        # print(istep, np.min(new_angles / np.pi*180), np.max(new_angles / np.pi*180))
        prj_dirs.append(new_angles)

    prj_dirs = np.concatenate(prj_dirs, 0)

    dangle = np.pi * 2 / len(angles)

    iphases = np.floor(prj_dirs / np.pi)
    warped_angles = prj_dirs - iphases * np.pi

    ind_sorted_angles = np.argsort(warped_angles)
    sorted_angles = warped_angles[ind_sorted_angles]

    # first angle
    w0 = (sorted_angles[1] - (sorted_angles[-1] - np.pi)) / dangle / 2

    # last angle
    wn = ((sorted_angles[0] + np.pi) - sorted_angles[-2]) / dangle / 2

    weights = [w0]
    for i in range(1, len(sorted_angles) - 1):
        weights.append((sorted_angles[i + 1] - sorted_angles[i - 1]) / dangle / 2)
    weights.append(wn)
    weights = np.array(weights)

    # convert weights back to the original order
    weights = weights[ind_sorted_angles.argsort()]


    num = sinogram_segments.shape[0]
    sinogram_weighted = np.zeros(sinogram_segments.shape)
    for i in range(0,num):
        s = segment_indexes[i]
        segment = sinogram_segments[i,...]
        current_weights = weights[s[0]: s[1]]
        sinogram_weighted[i,...] = segment * current_weights[np.newaxis, :, np.newaxis, np.newaxis]

    return sinogram_weighted, weights