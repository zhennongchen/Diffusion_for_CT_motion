import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy.random as random
import os
import math
import nibabel as nb
from scipy import interpolate
from Diffusion_for_CT_motion.simulation_data_generation.transformation.generate_transformation_matrix import *

# # def spline_fit(x,y):#
#     t, c, k = interpolate.splrep(x,y, s=0, k=3)
#     spline = interpolate.BSpline(t, c, k, extrapolate=False)
#     return spline, t, c, k

# cubic hermite spline
def h_poly_helper(tt0, tt1, tt2, tt3):
  A = np.array([
    [1, 0, -3, 2],
    [0, 1, -2, 1],
    [0, 0, 3, -2],
    [0, 0, -1, 1]], dtype=tt1[-1].dtype)

  first = A[0,0] * tt0  + A[0,1] * tt1 + A[0,2] * tt2 + A[0,3] * tt3
  second = A[1,0] * tt0  + A[1,1] * tt1 + A[1,2] * tt2 + A[1,3] * tt3
  third = A[2,0] * tt0  + A[2,1] * tt1 + A[2,2] * tt2 + A[2,3] * tt3
  fourth = A[3,0] * tt0  + A[3,1] * tt1 + A[3,2] * tt2 + A[3,3] * tt3
  output = np.asarray([first, second, third, fourth])

  return output

def h_poly(t):
  tt0 = 1
  tt1 = tt0 * t # same dimension as t
  tt2 = np.multiply(tt1, t) # same dimension as t
  tt3 = np.multiply(tt2, t)

  return h_poly_helper(tt0, tt1, tt2 , tt3)


def interp_func(x, y):
  m = (y[1:] - y[:-1])/(x[1:] - x[:-1])
  m = np.concatenate([m[[0]], (m[1:] + m[:-1])/2, m[[-1]]])
  def f(xs):

    I = np.searchsorted(x[1:], xs)  # each xs belong to which spline segment
    dx = (x[I+1]-x[I])

    hh = h_poly((xs-x[I])/dx)
    term1 = np.multiply(hh[0],y[I])
    term2 = np.multiply(np.multiply(hh[1], m[I]) ,dx)
    term3 = np.multiply(hh[2],y[I+1])
    term4 = np.multiply(np.multiply(hh[3],m[I+1]),dx)
    return term1 + term2 + term3 + term4
  return f

def interp(x, y, xs):
  return interp_func(x,y)(xs)


def split_into_direction(amplitude,dim):
    while True:
        split = np.zeros(dim)
        for i in range(0,dim-1):
            split[i] = np.random.rand()
        split[-1] = 1 - np.sum(split)
        if split[-1] >= 0:
            break
    split = [math.sqrt(i * (amplitude ** 2)) for i in split]
    return np.asarray(split)

def direciton_vector(num, change_direction_limit):
    # e.g. if change_direction_limit = 2, the direction can only change every change_direction_limit points
    a = [-1 if int(np.random.uniform(-1,1) >= 0) == 0 else 1 for i in (0,1)][0]

    v = np.ones(num) * a
    same_direction_count = 0
    for i in range(1,num):
        same_direction_count += 1
        if same_direction_count <= change_direction_limit: # no change
            v[i] = v[i-1]
        if same_direction_count > change_direction_limit: # have the chance to change
            v[i] = [-1 if int(np.random.uniform(-1,1) >= 0) == 0 else 1 for i in (0,1)][0]
            if v[i] != v[i-1]:
                same_direction_count = 1
    return v

def euclidean_distance(u, v):
    return np.sqrt(np.sum(np.power(u - v, 2)))


# define the function to make control points for both translation and rotation generation:
def motion_control_point_generation(dim, CP_num, amplitude_max, displacement_max, change_direction_limit, offset_value = None, print_result =False):
    if offset_value == None:
        offset = np.zeros(dim)
    else:
        offset = offset_value

    while True:
        t = np.zeros([CP_num,dim]) + offset
        # first randomly sample the displacmeent between two CPs  from [0,t_displacement_max]
        t_displacement= [np.random.rand() * displacement_max for i in range(0,CP_num - 1)]

        # get the direction vectors
        directions = np.ones([CP_num,dim]) 
        for i in range(0,dim):
            directions[:,i] = np.reshape(direciton_vector(CP_num, change_direction_limit), (CP_num,))
        directions[0,:] = np.ones(dim)

        # split displacement in x and y
        t_split = [split_into_direction(dis, dim) for dis in t_displacement]
        t_split = np.reshape(t_split, [CP_num-1,dim])
        t[1:,...] = t_split

        # assign direction to each control point
        t_direct = np.multiply(t, directions)

        # make amplitude vector
        t_amplitude = np.zeros([CP_num,dim])
        for i in range(0,dim):
            a = t_direct[:,i]
            a = [np.sum(a[0:i+1]) for i in range(0,CP_num)]
            t_amplitude[:,i] = a
    
        # check amplitude never exceed t_amplitude_max
        amplitude_euclidean = np.zeros(CP_num)
        for i in range(0,CP_num):
            amplitude_euclidean[i] = euclidean_distance(t_amplitude[i ,:] , offset)
        Assert = np.asarray([1 if amplitude_euclidean[i] > amplitude_max  else 0 for i in range(0,amplitude_euclidean.shape[0])])


        if np.sum(Assert) == 0:
            if print_result == True:
                # print('displacement: ',t_displacement)
                # print('directional displacement: ',t_direct)
                print('motion: ', np.reshape(t_amplitude,-1), '  max amplitude: ',np.max(amplitude_euclidean))
            
            return t_amplitude
