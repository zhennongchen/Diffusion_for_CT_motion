from fcntl import DN_MODIFY
import numpy as np
from numpy import random
import math
from .rotation_matrix_from_angle import rotation_matrix


def generate_transform_matrix(t,r,s,img_shape, which_one_is_first = 'rotation'):

    assert type(img_shape) == tuple
   
    ## translation
    # t should be the translation in [x,y,z] directions
    assert len(t) == len(img_shape)

    translation = np.eye(len(img_shape) + 1)
    translation[:len(img_shape),len(img_shape)] = np.transpose(np.asarray(t))

    ## rotation
    # r should be the rotation angle in [x,y,z] if 3D, or a single scalar value if 2D
    if len(img_shape) == 2:
        assert type(r) == float
    if len(img_shape) == 3:
        assert len(r) == 3
    
    rotation = np.eye(len(img_shape) + 1)

    if len(img_shape) == 2:

        rotation[:2, :2] = rotation_matrix(r)

    elif len(img_shape) == 3:
        x = rotation_matrix(r[0],matrix_type="roll")
        y = rotation_matrix(r[1],matrix_type="pitch")
        z = rotation_matrix(r[2],matrix_type="yaw")
        rotation[:3, :3] = np.dot(z, np.dot(y, x))
    else:
        raise Exception("image_dimension must be either 2 or 3.")

    # scale
    scale = np.eye(len(img_shape) + 1)
    for ax in range(0,len(img_shape)):
        scale[ax, ax] = s[ax]
    
    if which_one_is_first[0:2] == 'ro':
        return translation,rotation,scale,np.dot(scale, np.dot(rotation, translation))
    if which_one_is_first[0:2] == 'tr':
        return translation,rotation,scale,np.dot(scale, np.dot(translation, rotation))



def random_angle(input, do_sign = True):

    if do_sign == True:
        a = random.rand()
        if a >= 0.5:
            return random.rand() * input
        else:
            return random.rand() * -input
    else:
        return random.rand() * input


def random_t(max_t, ndim, do_sign = True):
    total_t = random.rand() * max_t
    
    # randomly divide into x and y (and z)
    proportion = []
    for i in range(0,ndim):
        proportion.append(random.rand())
    # final t
    final_t = []
    for i in range(0,ndim):
        tt = math.sqrt(total_t**2 / sum(proportion) * proportion[i])
        if do_sign == True:
            a = random.rand()
            if a >=0.5:
                tt = -tt
        final_t.append(tt)
    
    return final_t, total_t


def generate_random_2D_motion(max_t,max_rx, pixel_dim , time_range = [250,400],total_view_num = 2340, gantry_rotation_time = 500, no_random = False):
    # default, max_t (maximum translation) no more than 5mm， no translation in x (z axis in image)
    # default, max_rx (maximum rotation) no more than 5 degree. x axis is the original z axis in the image.
    # time_interval is the range of possible lasting time, a list with [minimum_lasting_time, maximum_lasting_time], default [200,400]ms
    # total_view_new = 2400 for conventional CT
    # gantry_rotation_time, usually 500 ms or 250 ms per gantry rotation

    per_view_time = gantry_rotation_time / total_view_num

    # get the randomized translation
    if no_random == False:
        final_t, _  = random_t(max_t,2)
        t_x, t_y, t_z = [0, final_t[0], final_t[1]]
        translation = [t_x/pixel_dim[0], t_y/pixel_dim[1], t_z/pixel_dim[-1]]
        translation_mm = [t_x,t_y,t_z]

        # get the randomized rotation
        rotation = [random_angle(max_rx) ,0,0]
        rotation = [i / 180 * np.pi for i in rotation]

    else:
        translation_mm = [0, max_t, max_t]
        translation = [0/pixel_dim[0], max_t/pixel_dim[1], max_t/pixel_dim[-1]]
        rotation = [max_rx, 0, 0]
        rotation = [i / 180 * np.pi for i in rotation]
    
    # get start view and end view
    # no motion situation:
    if sum(translation_mm) + sum(rotation) == 0:
        start_view = 0
        end_view = 0
        lasting_view = 0
        lasting_time = 0
    else:
        # get a randomized lasting time
        lasting_time = int(random.rand() * (time_range[1] - time_range[0]) + time_range[0])

        lasting_view = int(lasting_time / per_view_time)    

        start_view = int(random.rand() * (total_view_num - lasting_view - 1))
        end_view = int(start_view + lasting_view)
    
    return translation,translation_mm,rotation,start_view,end_view,lasting_view,lasting_time


   
#
# def generate_random_3D_motion(max_t,max_rx, max_ryz, pixel_dim , time_range = [250,400],total_view_num = 2340, gantry_rotation_time = 500, no_random = False):
#     # default, max_t (maximum translation) no more than 5mm
#     # default, max_rx (maximum rotation) no more than 15 degree ,max_ryz no more than 3 degree. x axis is the original z axis in the image.
#     # time_interval is the range of possible lasting time, a list with [minimum_lasting_time, maximum_lasting_time], default [200,400]ms
#     # total_view_new = 2400 for conventional CT
#     # gantry_rotation_time, usually 500 ms or 250 ms per gantry rotation

#     per_view_time = gantry_rotation_time / total_view_num

#     # get the randomized translation
#     if no_random == False:
#         # while True:
#         #     t_x,t_y,t_z = [random_num(max_t) ,random_num(max_t),random_num(max_t)]
            
#         #     if math.sqrt((t_x**2 + t_y**2 + t_z**2)) <= max_t:
#         #         break
#         t_x, t_y, t_z = random_t(max_t)
#         translation = [t_x/pixel_dim[0], t_y/pixel_dim[1], t_z/pixel_dim[-1]]
#         translation_mm = [t_x,t_y,t_z]

#         # get the randomized rotation
#         rotation = [random_angle(max_rx) ,random_angle(max_ryz),random_angle(max_ryz)]
#         rotation = [i / 180 * np.pi for i in rotation]

#     else:
#         translation_mm = [max_t, max_t, max_t]
#         translation = [max_t/pixel_dim[0], max_t/pixel_dim[1], max_t/pixel_dim[-1]]
#         rotation = [max_rx,max_ryz,max_ryz]
#         rotation = [i / 180 * np.pi for i in rotation]
    
#     # get start view and end view
#     # no motion situation:
#     if sum(translation_mm) + sum(rotation) == 0:
#         start_view = 0
#         end_view = 0
#         lasting_view = 0
#         lasting_time = 0
#     else:
#         # get a randomized lasting time
#         lasting_time = int(random.rand() * (time_range[1] - time_range[0]) + time_range[0])

#         lasting_view = int(lasting_time / per_view_time)

#         start_view = int(random.rand() * (total_view_num - lasting_view - 1))
#         end_view = int(start_view + lasting_view)
    
#     return translation,translation_mm,rotation,start_view,end_view,lasting_view,lasting_time
