#!/usr/bin/env python

import importlib
import CTProjector.src.ct_projector
importlib.reload(CTProjector.src.ct_projector)

import CTProjector.src.ct_projector.projector.cupy as ct_projector
import CTProjector.src.ct_projector.projector.cupy.fan_equiangular as ct_fan
import CTProjector.src.ct_projector.projector.numpy as numpy_projector
import CTProjector.src.ct_projector.projector.numpy.fan_equiangluar as numpy_fan
import CTProjector.src.ct_projector.projector.cupy.parallel as ct_para
import CTProjector.src.ct_projector.projector.numpy.parallel as numpy_para

import numpy as np
import cupy as cp
import os
import glob as gb
import nibabel as nb
import Diffusion_for_CT_motion.simulation_data_generation.functions as ff
import Diffusion_for_CT_motion.simulation_data_generation.transformation as transform



# for nibabel:
def basic_image_processing(filename , convert_value = True, header = False):
    ct = nb.load(filename)
    spacing = ct.header.get_zooms()
    img = ct.get_fdata()
    
    if convert_value == True:
        img = (img.astype(np.float32) + 1024) / 1000 * 0.019
        img[img < 0] = 0
        
    img = np.rollaxis(img,-1,0)

    spacing = np.array(spacing[::-1])

    if header == False:
        return img,spacing,ct.affine
    else:
        return img, spacing, ct.affine, ct.header


def define_forward_projector(img,spacing,total_view,du_nyquist = 1):
    projector = ct_projector.ct_projector()
    projector.from_file('./projector_fan.cfg')
    projector.nx = img.shape[3]
    projector.ny = img.shape[2]
    projector.nz = 1
    projector.nv = 1
    projector.dx = spacing[2]
    projector.dy = spacing[1]
    projector.dz = spacing[0]
    projector.nview = total_view
    if du_nyquist != 0:
        nyquist = projector.dx * projector.dsd / projector.dso / 2
        projector.du = nyquist * du_nyquist

    # for k in vars(projector):
    #     print (k, '=', getattr(projector, k))
    return projector


def backprojector(img,spacing, du_nyquist = 1):
    fbp_projector = numpy_projector.ct_projector()
    fbp_projector.from_file('./projector_fan.cfg')
    fbp_projector.nx = img.shape[3]
    fbp_projector.ny = img.shape[2]
    fbp_projector.nz = 1
    fbp_projector.nv = 1
    fbp_projector.dx = spacing[2]
    fbp_projector.dy = spacing[1]
    fbp_projector.dz = spacing[0]

    if du_nyquist != 0:
        nyquist = fbp_projector.dx * fbp_projector.dsd / fbp_projector.dso / 2
        fbp_projector.du = nyquist * du_nyquist

    return fbp_projector


def fp_static(img,angles,projector, geometry):
    # cp.cuda.Device(0).use()
    # ct_projector.set_device(0)
    origin_img = img[0, ...]
    origin_img = origin_img[:, np.newaxis, ...]
    cuimg = cp.array(origin_img, cp.float32, order = 'C')
    cuangles = cp.array(angles, cp.float32, order = 'C')

    if geometry[0:2] == 'fa':
        projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=False)
        numpy_projector.set_device(0)
    else:
        projector.set_projector(ct_para.distance_driven_fp, angles = cuangles,branchless = False)
        numpy_projector.set_device(0)

    # forward projection
    cufp = projector.fp(cuimg, angles = cuangles)
    fp = cufp.get()

    return fp



def fp_w_spline_motion_model(img, projector, angles, spline_tx, spline_ty, spline_tz, spline_rx, spline_ry, spline_rz, geometry,  total_view = 1400, gantry_rotation_time = 500, slice_num = None, increment = 28 , order = 3):
    if slice_num is None:
        slice_num = [0,img.shape[1]]
    projection = np.zeros([slice_num[1] - slice_num[0],angles.shape[0],1,projector.nu])
    view_to_time = gantry_rotation_time / total_view

    view = 0
    t_end_list = []
    while True:

        view_start = view
        view_end = view + increment

        t_start = view_start * view_to_time
        t_end = view_end * view_to_time
        t_end_list.append(t_end)
        # print('view: ',view_start, view_end, t_start, t_end)
        
        # if view_end % 100 == 0:
        #     print('view: ',view_start, '~',view_end, ' time: ', np.round(t_start,2), '~', np.round(t_end,2))
        
        # get the motion at this time point
        translation_ = [spline_tz(np.array([t_end])), spline_tx(np.array([t_end])), spline_ty(np.array([t_end]))]
        rotation_ = [spline_rz(np.array([t_end])), spline_rx(np.array([t_end])), spline_ry(np.array([t_end]))]

        # print('t: ', t_end, ' motion: ',translation_, [rr/np.pi*180 for rr in rotation_])

        I = img[0,...]
        _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],I.shape)
        transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
        img_new = transform.apply_affine_transform(I, transformation_matrix ,order)
        img_new = img_new[np.newaxis, ...]
                
        origin_img = img_new[0,slice_num[0]:slice_num[1],...]
        origin_img = origin_img[:, np.newaxis, ...]

        cuimg = cp.array(origin_img, cp.float32, order = 'C')
        cuangles = cp.array(angles[view_start : view_end], cp.float32, order = 'C')
        # print('angles: ', angles[view_start : view_end] / np.pi * 180)

        if geometry[0:2] == 'fa': # fan beam
            projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=False)
        elif geometry[0:2] == 'pa': # parallel beam:
            projector.set_projector(ct_para.distance_driven_fp, angles = cuangles)
        else:
            ValueError('wrong geometry')
        cufp = projector.fp(cuimg, angles = cuangles)

        fp = cufp.get()

        projection[:,view_start:view_end,...] = fp

        view =  view + increment
        if view >= angles.shape[0]:
            break
    
    return projection
    


def filtered_backporjection(projection,angles,projector,fbp_projector, geometry, back_to_original_value = True):
    # z_axis = True when z_axis is the slice, otherwise x-axis is the slice

    cuangles = cp.array(angles, cp.float32, order = 'C')
    if geometry[0:2] == 'fa':
        fprj = numpy_fan.ramp_filter(fbp_projector, projection, filter_type='RL')
        projector.set_backprojector(ct_fan.distance_driven_bp, angles=cuangles, is_fbp=True)
    elif geometry[0:2] == 'pa':
        fprj = numpy_para.ramp_filter(fbp_projector, projection, filter_type='RL')
        projector.set_backprojector(ct_para.distance_driven_bp, angles=cuangles, is_fbp=True)
    else:
        ValueError('wrong geometry')

    cufprj = cp.array(fprj, cp.float32, order = 'C')
    curecon = projector.bp(cufprj)
    recon = curecon.get()
    recon = recon[:,0,...]

    if back_to_original_value == True:
        recon = recon / 0.019 * 1000 - 1024

    return recon



##### PAR-related
def divide_sinogram_indexes_direct_num(segment_num , total_view_num):
    #sinogram dim should be [slice_num, view_num, 1, detector_num]

    num_angles_in_one_segment = int(round(total_view_num / segment_num))
    increment = num_angles_in_one_segment

    # find indexes for each segment
    segment_indexes = []
    for k in range(0,segment_num):
        # center = center_angle_index + k * increment
        s = [k * increment, k*increment + increment]
        if s[1] > total_view_num:
            s[1] = total_view_num
        segment_indexes.append(s)
    segment_indexes = np.asarray(segment_indexes)
    
    return segment_indexes,  num_angles_in_one_segment

def divide_sinogram_direct_num(sinogram, segment_num , total_view_num):
    #sinogram dim should be [slice_num, view_num, 1, detector_num]
    # assert sinogram.shape[1] == total_view_num

    segment_indexes, num_angles_in_one_segment = divide_sinogram_indexes_direct_num(segment_num, total_view_num)
    increment = num_angles_in_one_segment

    # make sinogram segments
    sinogram_segments = np.zeros((segment_indexes.shape[0], sinogram.shape[0], increment, sinogram.shape[2], sinogram.shape[3]))
    for i in range(0,sinogram_segments.shape[0]):
        s = segment_indexes[i]
        # print('segment index: ', i, ' slice: ', s[0], s[1])
        sinogram_segments[i,...] = sinogram[:,s[0] : s[1], :, :]

    return sinogram_segments, num_angles_in_one_segment, segment_indexes


def make_PAR_new(sinogram_segments, segment_indexes, angles, target_shape, projector, fbp_projector, geometry, back_to_original_value = True ):

    num = sinogram_segments.shape[0]
    final_par = np.zeros([num,target_shape[0], target_shape[1], target_shape[2]])

    for i in range(0,num):
        s = segment_indexes[i]
        segment = sinogram_segments[i,...]
        angles_partial = angles[s[0]:s[1]]

        # backprojection
        recon = filtered_backporjection(segment,angles_partial,projector,fbp_projector,geometry, back_to_original_value = back_to_original_value)
        final_par[i,:,:,:] = recon
   
    return final_par


def divide_sinogram_indexes(K , total_view_num, fill_out = False):
    #sinogram dim should be [slice_num, view_num, 1, detector_num]

    segment_num = 2*K + 1
    num_angles_in_one_segment = int(round(total_view_num / segment_num))
    increment = num_angles_in_one_segment
    center_angle_index = int(total_view_num / 2)

    # find indexes for each segment
    segment_indexes = []
    for k in range(-K,K+1):
        center = center_angle_index + k * increment
        s = [int(center - increment / 2), int(center + increment / 2)]
        if s[0] < 0:
            s[0] = 0
        if s[1] > total_view_num:
            s[1] = total_view_num
        
        if s[1] - s[0] != increment:
            if fill_out == False:
                continue
            else:
                if s[0] == 0:
                    s[1] = s[0] + increment
                if s[1] == total_view_num:
                    s[0] = total_view_num - increment

        segment_indexes.append(s)
    segment_indexes = np.asarray(segment_indexes)
    
    return segment_indexes, center_angle_index, num_angles_in_one_segment


def divide_sinogram_new(sinogram, K , total_view_num, fill_out = False):
    # sinogram dim should be [slice_num, view_num, 1, detector_num]
    # assert sinogram.shape[1] == total_view_num

    segment_indexes, center_angle_index, num_angles_in_one_segment = divide_sinogram_indexes(K, total_view_num, fill_out)
    increment = num_angles_in_one_segment

    # make sinogram segments
    sinogram_segments = np.zeros((segment_indexes.shape[0], sinogram.shape[0], increment, sinogram.shape[2], sinogram.shape[3]))
    for i in range(0,sinogram_segments.shape[0]):
        s = segment_indexes[i]
        sinogram_segments[i,...] = sinogram[:,s[0] : s[1], :, :]

    return sinogram_segments, center_angle_index, num_angles_in_one_segment, segment_indexes


def fp_w_delta_motion_model(img, projector, angles, spline_gt_tx, spline_gt_ty, spline_gt_tz, spline_gt_rx, spline_gt_ry, spline_gt_rz, spline_pred_tx, spline_pred_ty, spline_pred_tz, spline_pred_rx, spline_pred_ry, spline_pred_rz, geometry, total_view = 1440, gantry_rotation_time = 500, slice_num = None, increment = 5 , order = 3):
    if slice_num is None:
        slice_num = [0,img.shape[1]]

    projection = np.zeros([slice_num[1] - slice_num[0],angles.shape[0],1,projector.nu])
    view_to_time = gantry_rotation_time / total_view

    view = 0
    t_end_list = []
    while True:

        view_start = view
        view_end = view + increment

        # t_start = view_start * view_to_time
        t_end = view_end * view_to_time
        t_end_list.append(t_end)
        
        # get the motion at this time point
        translation_ = [spline_gt_tz(np.array([t_end])) - spline_pred_tz(np.array([t_end])), spline_gt_tx(np.array([t_end])) - spline_pred_tx(np.array([t_end])), spline_gt_ty(np.array([t_end])) - spline_pred_ty(np.array([t_end]))]
        rotation_ = [spline_gt_rz(np.array([t_end])) - spline_pred_rz(np.array([t_end])), spline_gt_rx(np.array([t_end])) - spline_pred_rx(np.array([t_end])), spline_gt_ry(np.array([t_end])) - spline_pred_ry(np.array([t_end]))]
        
        # print('t: ', t_end, ' motion: ',translation_, [rr/np.pi*180 for rr in rotation_])
    
        I = img[0,...]
        _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],I.shape)
        transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
        img_new = transform.apply_affine_transform(I, transformation_matrix ,order)
        img_new = img_new[np.newaxis, ...]
                
        origin_img = img_new[0,slice_num[0]:slice_num[1],...]
        origin_img = origin_img[:, np.newaxis, ...]

        cuimg = cp.array(origin_img, cp.float32, order = 'C')
        cuangles = cp.array(angles[view_start : view_end], cp.float32, order = 'C')
        # print('angles: ', angles[view_start : view_end] / np.pi * 180)

        if geometry[0:2] == 'fa': # fan beam
            projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=False)
        elif geometry[0:2] == 'pa': # parallel beam:
            projector.set_projector(ct_para.distance_driven_fp, angles = cuangles)
        else:
            ValueError('wrong geometry')
        cufp = projector.fp(cuimg, angles = cuangles)

        fp = cufp.get()

        projection[:,view_start:view_end,...] = fp

        view =  view + increment
        if view >= angles.shape[0]:
            break
    
    return projection


# def fp_w_linear_motion(img, projector, angles, translation, rotation, start_view, end_view,slice_num, increment_raw, geometry, order = 3):
#     if slice_num is None:
#         slice_num = [0,img.shape[1]]

#     projection = np.zeros([slice_num[1] - slice_num[0],angles.shape[0],1,projector.nu])

#     t = 0
#     transformation_doing = False
#     transformation_done = False
#     while True:
#         if t + increment_raw >= start_view and transformation_doing == False and transformation_done == False:
#             increment = start_view - t
#             transformation_doing = True
#         elif t + increment_raw >= end_view and transformation_doing == True and transformation_done == False:
#             increment = end_view - t
#             transformation_done = True
#         elif t + increment_raw > angles.shape[0] and transformation_done == True:
#             increment = angles.shape[0] - t
#         else:
#             increment = increment_raw

#         if start_view == 0 and end_view == 0: # no motion:
#             increment = increment_raw


#         if t < start_view:
#             img_new = np.copy(img)
#             print('view: ',t, '~',t+increment, ' keep original img')

#         elif t >= start_view and t<end_view:
#             translation_ = [i / (end_view - start_view) * (t - start_view) for i in translation]
#             rotation_ = [i / (end_view - start_view) * (t - start_view) for i in rotation]
        
#             I = img[0,...]
#             _,_,_,transformation_matrix = transform.generate_transform_matrix(translation_,rotation_,[1,1,1],I.shape)
#             transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
#             img_new = transform.apply_affine_transform(I, transformation_matrix , order)
#             img_new = img_new[np.newaxis, ...]

#             print('view: ',t, '~',t+increment, ' doing transformation with translation: ', translation_, ' rotation: ',[rr/np.pi*180 for rr in rotation_])
                
#         else:
#             I = img[0,...]
#             _,_,_,transformation_matrix = transform.generate_transform_matrix(translation,rotation,[1,1,1],I.shape)
#             transformation_matrix = transform.transform_full_matrix_offset_center(transformation_matrix, I.shape)
#             img_new = transform.apply_affine_transform(I, transformation_matrix, order)
#             img_new = img_new[np.newaxis, ...]
#             print('view: ',t, '~',t+increment,' after transformation')

#         origin_img = img_new[0,slice_num[0]:slice_num[1],...]
#         origin_img = origin_img[:, np.newaxis, ...]

#         cuimg = cp.array(origin_img, cp.float32, order = 'C')
#         cuangles = cp.array(angles[t:t+increment], cp.float32, order = 'C')

#         if geometry[0:2] == 'fa': # fan beam
#             projector.set_projector(ct_fan.distance_driven_fp, angles=cuangles, branchless=False)
#         elif geometry[0:2] == 'pa': # parallel beam:
#             projector.set_projector(ct_para.distance_driven_fp, angles = cuangles)
#         else:
#             ValueError('wrong geometry')

#         cufp = projector.fp(cuimg, angles = cuangles)

#         fp = cufp.get()

#         projection[:,t:t+increment,...] = fp

#         t = t+increment
#         if t >= angles.shape[0]:
#             break
    
#     return projection

# def divide_sinogram(sinogram, K , total_view_num):
#     #sinogram dim should be [slice_num, view_num, 1, detector_num]
#     # assert sinogram.shape[1] == total_view_num

#     segment_num = 2*K + 1
#     num_angles_in_one_segment = int(total_view_num / segment_num)
#     increment = num_angles_in_one_segment
#     center_angle_index = int(total_view_num / 2)

#     sinogram_segments = np.zeros((segment_num, sinogram.shape[0], increment, sinogram.shape[2], sinogram.shape[3]))

#     segment_indexes = []
#     for k in range(-K,K+1):
#         center = center_angle_index + k * increment
#         s = [int(center - increment / 2), int(center + increment / 2)]
#         segment_indexes.append(s)
#         sinogram_segments[k,...] = sinogram[:,s[0]:s[1],:,:]
#         # print(k,center,s)

#     return sinogram_segments, center_angle_index, num_angles_in_one_segment, np.asarray(segment_indexes)


# def make_PAR(sinogram_segments, K, center_angle_index, num_angles_in_one_segment, angles, target_shape, projector, fbp_projector, geometry ):
#     assert sinogram_segments.shape[0] == (2*K + 1)

#     num = 2*K + 1
#     final_par = np.zeros([num,target_shape[0], target_shape[1], target_shape[2]])
#     increment = num_angles_in_one_segment

#     for i in range(0,num):
#         k = i - K
#         center = center_angle_index + (k) * increment
#         s = [int(center - increment / 2), int(center + increment / 2)]

#         segment = sinogram_segments[k,...]
#         angles_partial = angles[s[0]:s[1]]
#         # print(k,center,s, np.sum(segment[0,...]), angles_partial)

#         # backprojection
#         recon = filtered_backporjection(segment,angles_partial,projector,fbp_projector,geometry, back_to_original_value = True)
#         final_par[i,:,:,:] = recon

#     return final_par