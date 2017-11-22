from __future__ import print_function, absolute_import, division

import gpu_config
import numpy as np
import tensorflow as tf
import data.util
from data.util import xyz2uvd_op, uvd2xyz_op, heatmap_from_xyz_op, CameraConfig
FLAGS = tf.app.flags.FLAGS

def crop_from_xyz_pose(dm, pose, cfg, out_w, out_h, pad=20.0):
    '''crop depth map by generate the bounding box according to the pose
    Args:
        dms: depth map
        poses: either estimated or groundtruth in xyz coordinate
        cfg: the initial camera configuration
        out_w: output width
        out_h: output height
    Returns:
        crop_dm: the cropped depth map
        new_cfg: the new camera configuration for the cropped depth map
    '''
    with tf.name_scope('crop'):
        # determine bouding box from pose
        in_h, in_w = dm.get_shape()[0].value, dm.get_shape()[1].value
        uvd_pose = tf.reshape(xyz2uvd_op(pose,cfg), (-1,3))
        min_coor = tf.reduce_min(uvd_pose, axis=0)
        max_coor = tf.reduce_max(uvd_pose, axis=0)

        top = tf.minimum(tf.maximum(min_coor[1]-pad, 0.0), cfg.h-2*pad)
        left = tf.minimum(tf.maximum(min_coor[0]-pad, 0.0), cfg.w-2*pad)
        bottom = tf.maximum(tf.minimum(max_coor[1]+pad, cfg.h), tf.cast(top, tf.float32)+2*pad-1)
        right = tf.maximum(tf.minimum(max_coor[0]+pad, cfg.w), tf.cast(left, tf.float32)+2*pad-1)

        top = tf.cast(top, tf.int32)
        left = tf.cast(left, tf.int32)
        bottom = tf.cast(bottom, tf.int32)
        right = tf.cast(right, tf.int32)

        cropped_dm = tf.image.crop_to_bounding_box(dm,
                                                  offset_height=top,
                                                  offset_width=left,
                                                  target_height=bottom-top,
                                                  target_width=right-left)

        longer_edge = tf.maximum(bottom-top, right-left)
        offset_height = tf.to_int32(tf.divide(longer_edge-bottom+top, 2))
        offset_width = tf.to_int32(tf.divide(longer_edge-right+left, 2))
        cropped_dm = tf.image.pad_to_bounding_box(cropped_dm,
                                                 offset_height=offset_height,
                                                 offset_width=offset_width,
                                                 target_height=longer_edge,
                                                 target_width=longer_edge)
        cropped_dm = tf.image.resize_images(cropped_dm, (out_h, out_w))

        # to further earse the background
        uvd_list = tf.unstack(uvd_pose, axis=-1)

        uu = tf.clip_by_value(tf.to_int32(uvd_list[0]), 0, in_w-1)
        vv = tf.clip_by_value(tf.to_int32(uvd_list[1]), 0, in_h-1)

        dd = tf.gather_nd(dm, tf.stack([vv,uu], axis=-1))
        dd = tf.boolean_mask(dd, tf.greater(dd, 100))
        d_th = tf.reduce_min(dd) + 250.0
        if FLAGS.dataset == 'icvl':
            cropped_dm = tf.where(tf.less(cropped_dm,500.0), cropped_dm, tf.zeros_like(cropped_dm))
        else:
            cropped_dm = tf.where(tf.less(cropped_dm,d_th), cropped_dm, tf.zeros_like(cropped_dm))

    with tf.name_scope('cfg'):
        ratio_x = tf.cast(longer_edge/out_w, tf.float32)
        ratio_y = tf.cast(longer_edge/out_h, tf.float32)
        top = tf.cast(top, tf.float32)
        left = tf.cast(left, tf.float32)

        new_cfg = tf.stack([cfg.fx/ratio_x, cfg.fy/ratio_y, 
                            (cfg.cx-left+tf.to_float(offset_width))/ratio_x, 
                            (cfg.cy-top+tf.to_float(offset_height))/ratio_y,
                            tf.cast(out_w,tf.float32), tf.cast(out_h,tf.float32)], axis=0) 
    return [cropped_dm, pose, new_cfg]

def crop_from_bbx(dm, pose, bbx, cfg, out_w, out_h):
    '''crop depth map by generate the bounding box according to the pose
    Args:
        dms: depth map
        pose: groundtruth pose for further error evaluation
        bbx: bounding box 
        cfg: the initial camera configuration
        out_w: output width
        out_h: output height
    Returns:
        crop_dm: the cropped depth map
        new_cfg: the new camera configuration for the cropped depth map
    '''
    with tf.name_scope('crop'):
        top, left, bottom, right, d_th = bbx[0], bbx[1], bbx[2], bbx[3], bbx[4]

        top = tf.cast(top, tf.int32)
        left = tf.cast(left, tf.int32)
        bottom = tf.cast(bottom, tf.int32)
        right = tf.cast(right, tf.int32)

        cropped_dm = tf.image.crop_to_bounding_box(dm,
                                                  offset_height=top,
                                                  offset_width=left,
                                                  target_height=bottom-top,
                                                  target_width=right-left)

        longer_edge = tf.maximum(bottom-top, right-left)
        offset_height = tf.to_int32(tf.divide(longer_edge-bottom+top, 2))
        offset_width = tf.to_int32(tf.divide(longer_edge-right+left, 2))
        cropped_dm = tf.image.pad_to_bounding_box(cropped_dm,
                                                 offset_height=offset_height,
                                                 offset_width=offset_width,
                                                 target_height=longer_edge,
                                                 target_width=longer_edge)
        cropped_dm = tf.image.resize_images(cropped_dm, (out_h, out_w))
        cropped_dm = tf.where(tf.less(cropped_dm,d_th), cropped_dm, tf.zeros_like(cropped_dm))

    with tf.name_scope('cfg'):
        ratio_x = tf.cast(longer_edge/out_w, tf.float32)
        ratio_y = tf.cast(longer_edge/out_h, tf.float32)
        top = tf.cast(top, tf.float32)
        left = tf.cast(left, tf.float32)

        new_cfg = tf.stack([cfg.fx/ratio_x, cfg.fy/ratio_y, 
                            (cfg.cx-left+tf.to_float(offset_width))/ratio_x, 
                            (cfg.cy-top+tf.to_float(offset_height))/ratio_y,
                            tf.cast(out_w,tf.float32), tf.cast(out_h,tf.float32)], axis=0) 
    return [cropped_dm, pose, new_cfg]

def center_of_mass(dm, cfg):
    shape = tf.shape(dm)
    c_h, c_w = shape[0], shape[1]
    ave_u, ave_v = tf.cast(c_w/2, tf.float32), tf.cast(c_h/2, tf.float32)
    ave_d = tf.reduce_mean(tf.boolean_mask(dm, tf.greater(dm,0)))

    ave_d = tf.maximum(ave_d, 200.0)

    ave_x = (ave_u-cfg[2])*ave_d/cfg[0]
    ave_y = (ave_v-cfg[3])*ave_d/cfg[1]
    ave_xyz=tf.stack([ave_x,ave_y,ave_d],axis=0)
    return ave_xyz

def norm_xyz_pose(xyz_poses, coms, pca_para=None):
    jnt_num = int(xyz_poses.shape[1].value/3) 
    def fn(elems):
        xyz_pose, com = elems[0], elems[1]
        norm_xyz_pose = tf.divide(xyz_pose - tf.tile(com,[jnt_num]), POSE_NORM_RATIO)
        return [norm_xyz_pose, com]

    norm_xyz_poses, _ = tf.map_fn(fn, [xyz_poses,coms])
    if pca_para is not None:
        norm_xyz_poses = tf.nn.xw_plus_b(norm_xyz_poses, tf.transpose(pca_para[0]), pca_para[2]) 
        norm_xyz_poses = tf.divide(norm_xyz_poses, PCA_NORM_RATIO)
    return norm_xyz_poses

def unnorm_xyz_pose(norm_xyz_poses, coms, pca_para=None):
    if pca_para is not None:
        norm_xyz_poses = tf.multiply(norm_xyz_poses, PCA_NORM_RATIO)
        norm_xyz_poses = tf.nn.xw_plus_b(norm_xyz_poses, pca_para[0], pca_para[1])
        # norm_xyz_poses = tf.matmul(norm_xyz_poses, pca_para[0])

    jnt_num = int(norm_xyz_poses.shape[1].value/3) 
    def fn(elems):
        norm_xyz_pose, com = elems[0], elems[1]
        xyz_pose = tf.multiply(norm_xyz_pose, POSE_NORM_RATIO) + tf.tile(com,[jnt_num])
        return [xyz_pose, com]

    xyz_poses, _ = tf.map_fn(fn, [norm_xyz_poses,coms])
    return xyz_poses

D_RANGE=300.0
POSE_NORM_RATIO = 100.0
PCA_NORM_RATIO  = 5.0

def norm_dm(dms, coms):
    def fn(elems):
        dm, com = elems[0], elems[1]
        max_depth = com[2]+D_RANGE*0.5
        min_depth = com[2]-D_RANGE*0.5
        mask = tf.logical_and(tf.less(dm, max_depth), tf.greater(dm, min_depth-D_RANGE*0.5))
        normed_dm = tf.where(mask, tf.divide(dm-min_depth, D_RANGE), -1.0*tf.ones_like(dm))
        return [normed_dm, com]

    norm_dms, _ = tf.map_fn(fn, [dms, coms])

    return norm_dms

def generate_xyzs_from_multi_cfgs(dms, cfgs, coms):
    '''generate the point cloud from depth map
    Args:
        dms: the normalized depth map, (b,h,w,1)
        cfgs: the corresponding camera configurations, (b, 6)
        coms: the corresponding center of mass, (b, 3)
    Returns:
        xyzs: the normalized xyz point cloud, (b, h, w, 3)
    '''

    def fn(elem):
        dm, cfg, com = elem[0], elem[1], elem[2]

        zz = tf.squeeze(dm, axis=-1)
        min_depth = com[2]-D_RANGE*0.5
        max_depth = com[2]+D_RANGE*0.5
        zz = tf.where(tf.less(zz, -0.99), 
                      tf.ones_like(zz)*max_depth, 
                      tf.multiply(zz, D_RANGE)+min_depth)

        xx, yy = tf.meshgrid(tf.range(h), tf.range(w))
        xx = tf.to_float(xx)
        yy = tf.to_float(yy)

        w_ratio = cfg[4]/w
        h_ratio = cfg[5]/h
        new_cfg = CameraConfig(cfg[0]/w_ratio, cfg[1]/h_ratio,
                               cfg[2]/w_ratio, cfg[3]/h_ratio,
                               w, h)

        xx = tf.multiply(xx-new_cfg[2], tf.divide(zz, new_cfg[0]))
        yy = tf.multiply(yy-new_cfg[3], tf.divide(zz, new_cfg[1]))

        # renormalize the points as normalizing the pose
        xx = tf.divide(xx-com[0], POSE_NORM_RATIO)
        yy = tf.divide(yy-com[1], POSE_NORM_RATIO)
        zz = tf.divide(zz-com[2], POSE_NORM_RATIO)

        xyz = tf.stack([xx,yy,zz], axis=-1)
        return [xyz, cfg, com]

    h, w = dms.get_shape()[1].value, dms.get_shape()[2].value
    xyzs, _, _ = tf.map_fn(fn, [dms, cfgs, coms])
    return xyzs

def data_aug(dms, poses, cfgs, coms):
    def fn(elems):
        dm, pose, cfg, com = elems[0], elems[1], elems[2], elems[3]
        # random rotation
        angle = tf.random_uniform((1,),-1*np.pi,np.pi)
        rot_dm = tf.contrib.image.rotate(dm,angle)
        
        uv_com = xyz2uvd_op(com, cfg)
        uvd_pt = xyz2uvd_op(pose, cfg) - tf.tile(uv_com,[jnt_num])
        cost, sint = tf.cos(angle)[0], tf.sin(angle)[0]
        rot_mat = tf.stack([cost,-sint,0, sint,cost,0, 0.0,0.0,1.0], axis=0)
        rot_mat = tf.reshape(rot_mat, (3,3))

        uvd_pt = tf.reshape(uvd_pt, (-1,3))
        rot_pose = tf.reshape(tf.matmul(uvd_pt, rot_mat), (-1,))
       
        # random elongate x,y edge
        edge_ratio = tf.clip_by_value(tf.random_normal((2,), 1.0, 0.2), 0.9, 1.1)
        target_height = tf.to_int32(tf.to_float(tf.shape(dm)[0])*edge_ratio[0])
        target_width = tf.to_int32(tf.to_float(tf.shape(dm)[1])*edge_ratio[1])
        # 1 stands for nearest neighour interpolation
        rot_dm = tf.image.resize_images(rot_dm, (target_height, target_width), 1)
        rot_dm = tf.image.resize_image_with_crop_or_pad(rot_dm, tf.shape(dm)[0], tf.shape(dm)[1])
        rot_pose = tf.multiply(rot_pose, tf.tile([edge_ratio[1],edge_ratio[0],1.0], [jnt_num]))

        rot_pose = rot_pose + tf.tile(uv_com, [jnt_num])
        rot_pose = uvd2xyz_op(rot_pose, cfg)
        rot_pose = tf.reshape(rot_pose, (-1,))
        return [rot_dm, rot_pose, cfgs, coms]

    jnt_num = tf.to_int32(tf.shape(poses)[1]/3)
    aug_dms, aug_poses, _, _ = tf.map_fn(fn, [dms, poses, cfgs, coms])
    aug_dms = tf.reshape(aug_dms, tf.shape(dms))
    return aug_dms, aug_poses

