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
import Diffusion_for_CT_motion.functions_collection as ff
import Diffusion_for_CT_motion.motion_simulation.transformation as transform
import glob as gb
import nibabel as nb


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
