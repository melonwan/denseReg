'''to simultaneously regress the 3D joint offset and the 2D joint heatmap
'''
from __future__ import print_function,absolute_import, division

import time, os
import numpy as np
from datetime import datetime
import importlib

import gpu_config
import tensorflow as tf
import data.util
from data.util import heatmap_from_xyz_op, CameraConfig, xyz2uvd_op, uvd2xyz_op
import data.util
import data.preprocess
import numpy as np, numpy.linalg as alg

# from model_new.train_single_gpu import train
from model.train_single_gpu import train
from model.test_model import test
import network.slim as slim

from data.preprocess import generate_xyzs_from_multi_cfgs, crop_from_xyz_pose, crop_from_bbx, center_of_mass, norm_xyz_pose, unnorm_xyz_pose

from data.visualization import tf_heatmap_wrapper, tf_jointplot_wrapper, tf_smppt_wrapper
from data.evaluation import Evaluation

# implementation setting
tf.app.flags.DEFINE_integer('num_gpus', 1, #gpu_config.num_gpus,
                           """how many gpu to be used""")
# use cpu instead if no gpu is available
tf.app.flags.DEFINE_integer('batch_size', 40,
                           '''batch size''')
tf.app.flags.DEFINE_integer('debug_level', 1,
                            '''the higher, the more saved to summary''')
tf.app.flags.DEFINE_integer('sub_batch', 5,
                           '''batch size''')
tf.app.flags.DEFINE_integer('pid', 0,
                           '''for msra person id''')
tf.app.flags.DEFINE_boolean('is_train', True,
                            '''True for traning, False for testing''')

# the network architecture to be used
tf.app.flags.DEFINE_string('net_module', 'um_v1',
                          '''the module containing the network architecture''')
tf.app.flags.DEFINE_boolean('is_aug', True,
                            '''whether to augment data''')
tf.app.flags.DEFINE_string('dataset', 'nyu',
                           '''the dataset to conduct experiments''')
# epoch
tf.app.flags.DEFINE_integer('epoch', 80,
                            '''number of epoches''')

# network specification
tf.app.flags.DEFINE_integer('num_stack', 2,
                            'number of stacked hourglass')
tf.app.flags.DEFINE_integer('num_fea', 128,
                            'number of feature maps in hourglass')
tf.app.flags.DEFINE_integer('kernel_size', 3,
                            'kernel size for the residual module')

FLAGS = tf.app.flags.FLAGS

MAXIMUM_DEPTH = 600.0

class JointDetectionModel(object):
    _moving_average_decay = 0.9999
    _batchnorm_moving_average_decay = 0.9997
    _init_lr = 0.001 
    if FLAGS.dataset == 'nyu':
        _num_epochs_per_decay = 10
    elif FLAGS.dataset == 'msra':
        _num_epochs_per_decay = 20 
    _lr_decay_factor = 0.1

    _adam_beta1 = 0.5
    
    # maximum allowed depth
    _max_depth = 600.0

    # input size: the input of the network
    _input_height = 128 
    _input_width = 128
    
    # output size: the output size of network, as well as the largest size of hourglass model
    _output_height = 32 
    _output_width = 32

    _gau_sigma = 3.0 
    _gau_filter_size = 10 

    _base_dir = './exp/train_cache/'


    def __init__(self, dataset, detect_net, epoch, net_desc='dummy', val_dataset=None):
        '''
        args:
            dataset: data.xxxdataset isinstance
            detect_net: funtional input of the net
            desc: string, the name of the corresponding cache folder 
        notice:
            any tf operations on the graph cannot be defined here,
            they can only be defined after the graph is initialized by the training module
        '''
        self._dataset = dataset
        self._jnt_num = int(dataset.jnt_num)
        self._cfg = self._dataset.cfg

        self._num_batches_per_epoch = dataset.approximate_num / (FLAGS.batch_size*FLAGS.sub_batch)
        self._net_desc = net_desc
        self._net = detect_net
        self._max_steps = int(epoch*self._num_batches_per_epoch)

        self._val_dataset = val_dataset 
        self._model_desc = '%s_%s_s%d_f%d'%(dataset.name, dataset.subset, FLAGS.num_stack, FLAGS.num_fea)
        if FLAGS.is_aug:
            self._model_desc += '_daug'

        if self._val_dataset:
            assert self._jnt_num == self._val_dataset.jnt_num, (
                'the validation dataset should be with the same number of joints to the traning dataset') 

        if not os.path.exists(self._base_dir):
            os.makedirs(self._base_dir)

        self._log_path = os.path.join(self._base_dir, self.name, 'validation_log.txt')

    '''data interface
    1. initialize the dataset
    2. the global setting of the batch_size
    3. total number of steps
    '''
    def batch_input(self, dataset, batch_size=None):
        if batch_size is None:
            batch_size = FLAGS.batch_size
        dm_batch, pose_batch, cfg_batch, com_batch = dataset.get_batch_op(
            batch_size=batch_size,
            num_readers = 2,
            num_preprocess_threads = 2,
            preprocess_op=dataset.preprocess_op(self._input_width, self._input_height))
        return [dm_batch, pose_batch, cfg_batch, com_batch]

    def batch_input_test(self, dataset):
        dm_batch, pose_batch, cfg_batch, com_batch, name_batch = dataset.get_batch_op_test(
            batch_size = FLAGS.batch_size, 
            preprocess_op = dataset.preprocess_op(self._input_width, self._input_height))
        return [dm_batch, pose_batch, cfg_batch, com_batch, name_batch]

    @property
    def train_dataset(self):
        return self._dataset

    @property
    def val_dataset(self):
        return self._val_dataset

    '''hyper parameters
    '''
    @property
    def init_lr(self):
        '''the initial learning rate
        '''
        return self._init_lr
    @property
    def lr_decay_factor(self):
        '''the rate of exponential decay of learning rate
        '''
        return self._lr_decay_factor

    @property
    def decay_steps(self):
        '''lr does not decay when global_step < decay_steps
        '''
        return self._num_batches_per_epoch * self._num_epochs_per_decay

    @property
    def moving_average_decay(self):
        return self._moving_average_decay

    @property
    def max_steps(self):
        return self._max_steps

    '''training operation
    '''
    def inference(self, normed_dms, cfgs, coms, reuse_variables, is_training=True):
        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
            # resize the image to fit the network input
            # during training, inference is called by loss function, where dms are resized
            end_points = self._net(normed_dms, cfgs, coms, self._jnt_num, is_training)
            return end_points 

    max_dist_2d = 4.0 # 4 pixels
    max_dist_3d = 0.8 # 80mm 3d distance 
    def _hm_3d(self, oms):
        '''generate 3D distance heatmap according to the offset map
        Args:
            oms: the normalized xyz offset maps, (b,h,w,3*j)
        Returns:
            hms: the 3D heatmap, (b,h,w,j)
        '''
        om_list = tf.unstack(oms, axis=-1)
        hm_list = []
        for j in range(self._jnt_num):
            xx,yy,zz = om_list[j*3], om_list[j*3+1], om_list[j*3+2]
            hm = tf.sqrt(xx**2+yy**2+zz**2)
            hm = tf.divide(self.max_dist_3d-hm, self.max_dist_3d)
            hm = tf.maximum(hm, tf.zeros_like(hm))
            hm_list.append(hm)
        hms = tf.stack(hm_list, axis=-1)
        return hms

    def _hm_2d(self, poses, cfgs, out_h, out_w):
        '''synthesize the 2d heatmap
        Args:
            poses: unnormed xyz pose, (b,j*3)
            cfgs: camera configuration, (b, 6)
            out_h, out_w: output of heatmap size
        Returns:
            hm2: 2D heatmap, (b, out_h, out_w, j)
        '''
        def fn(elems):
            xyz_pose, cfg = elems[0], elems[1]

            w_ratio = cfg[4] / out_w
            h_ratio = cfg[5] / out_h
            new_cfg = CameraConfig(cfg[0]/w_ratio, cfg[1]/h_ratio,
                                   cfg[2]/w_ratio, cfg[3]/h_ratio,
                                   out_w, out_h)

            xx, yy = tf.meshgrid(tf.range(out_h), tf.range(out_w))
            xx, yy = tf.cast(xx, tf.float32), tf.cast(yy, tf.float32)
            xx = tf.tile(tf.expand_dims(xx, axis=-1), [1, 1, self._jnt_num])
            yy = tf.tile(tf.expand_dims(yy, axis=-1), [1, 1, self._jnt_num])

            uvd_pose = tf.reshape(data.util.xyz2uvd_op(xyz_pose, new_cfg), (-1,3))
            [uu,vv,dd] = tf.unstack(uvd_pose, axis=-1)
            uu = tf.reshape(uu, (1,1,-1))
            vv = tf.reshape(vv, (1,1,-1))
            
            hm = tf.maximum(self.max_dist_2d-tf.sqrt(tf.square(xx-uu)+tf.square(yy-vv)),
                            tf.zeros_like(xx))/self.max_dist_2d
            return [hm, cfg]

        with tf.name_scope('pose_sync'):
            hms, _ = tf.map_fn(fn, [poses, cfgs])
            return hms

    def _um(self, om, hm_3d):
        '''get the unit offset vector map from offset maps
        Args:
            om: the offset map, (b,h,w,j*3)
            hm_3d: the offset norm, (b,h,w,j)
        Returns:
            um: the unit offset map, (b,h,w,j*3)
        '''
        om_list = tf.unstack(om, axis=-1)

        dm_3d = self.max_dist_3d - tf.multiply(hm_3d, self.max_dist_3d)
        dm_list = tf.unstack(dm_3d, axis=-1)

        um_list = []

        for j in range(self._jnt_num):
            x,y,z = om_list[j*3], om_list[j*3+1], om_list[j*3+2]
            d = dm_list[j]

            mask = tf.less(d, self.max_dist_3d-1e-2) 

            x = tf.where(mask, tf.divide(x, d), tf.zeros_like(x))
            y = tf.where(mask, tf.divide(y, d), tf.zeros_like(y))
            z = tf.where(mask, tf.divide(z, d), tf.zeros_like(z))
            um_list += [x,y,z]
        return tf.stack(um_list, axis=-1)

    def _resume_om(self, hm_3d, um):
        '''resume the offset map from the 3d heatmap and unit offset vector
        Args:
            hm_3d: the 3D heatmap, (b,h,w,j)
            um: the 3D unit offset vector, (b,h,w,j*3)
        Returns:
            om: the 3D offset vector, (b,h,w,j)
        '''
        # um = tf.clip_by_value(um, -1.0, 1.0)
        um_list = tf.unstack(um, axis=-1)

        dm_3d = self.max_dist_3d - tf.multiply(hm_3d, self.max_dist_3d)
        dm_list = tf.unstack(dm_3d, axis=-1)

        om_list = []

        for j in range(self._jnt_num):
            x,y,z = um_list[j*3], um_list[j*3+1], um_list[j*3+2]
            d = dm_list[j]
            x = tf.multiply(x,d)
            y = tf.multiply(y,d)
            z = tf.multiply(z,d)
            om_list += [x,y,z]
        return tf.stack(om_list, axis=-1)

    def _vis_um_xy(self, ums):
        '''visualize the xy plane angle of ums
        '''
        um_list = tf.unstack(ums, axis=-1)
        angle_list = []
        for j in range(self._jnt_num):
            x,y,z = um_list[j*3], um_list[j*3+1], um_list[j*3+2]
            d = tf.sqrt(x**2+y**2)
            sin = tf.where(tf.less(d**2+z**2, 0.1), tf.ones_like(d), tf.sin(tf.divide(x,d)))
            angle_list.append(sin)
        return tf.stack(angle_list, axis=-1)

    def _vis_um_z(self, ums):
        '''visuzlie the z plane angle of ums
        '''
        um_list = tf.unstack(ums, axis=-1)
        angle_list = []
        for j in range(self._jnt_num):
            angle_list.append(um_list[j*3+2])
        return tf.stack(angle_list, axis=-1)

    # training
    def loss(self, dms, poses, cfgs, coms):
        ''' the losses for the training
        Args:
            dms:
            poses:
            reuse_variables:
        Returns:
            the total loss 
        '''
        if FLAGS.is_aug:
            dms, poses = data.preprocess.data_aug(dms, poses, cfgs, coms)

        # generate ground truth
        gt_hms = self._hm_2d(poses, cfgs, self._output_height, self._output_width)

        gt_normed_poses = norm_xyz_pose(poses, coms) 
        normed_dms = data.preprocess.norm_dm(dms, coms)
        tiny_normed_dms = tf.image.resize_images(normed_dms, (self._output_height, self._output_width), 2)
        xyzs = generate_xyzs_from_multi_cfgs(tiny_normed_dms, cfgs, coms)
        xyzs = tf.tile(xyzs, [1,1,1,self._jnt_num])
        gt_oms = tf.reshape(gt_normed_poses, (-1,1,1,3*self._jnt_num)) - xyzs

        gt_hm3s = self._hm_3d(gt_oms)
        gt_ums = self._um(gt_oms, gt_hm3s)

        # generate estimation
        end_points = self.inference(normed_dms, cfgs, coms, reuse_variables=None, is_training=True)
        
        # heatmap loss
        est_hm_list = end_points['hm_outs'] 
        hm_losses = [tf.nn.l2_loss(est_hms-gt_hms) for est_hms in est_hm_list]

        # 3D heatmap loss
        est_hm3_list = end_points['hm3_outs']
        hm3_losses = [tf.nn.l2_loss(est_hm3-gt_hm3s) for est_hm3 in est_hm3_list]

        # offsetmap loss
        # we only consider the nearby point offset maps
        # in order to make the oms loss on the same scale w.r.t. hms loss
        est_um_list = end_points['um_outs'] 
        um_losses = [tf.nn.l2_loss(est_ums-gt_ums) for est_ums in est_um_list]

        # add the weight decay loss
        reg_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), 'reg_loss')
        hm_loss = tf.add_n(hm_losses, 'hm_loss')
        um_loss = tf.add_n(um_losses, 'um_loss')
        hm3_loss = tf.add_n(hm3_losses, 'hm3_loss')

        total_loss = reg_loss+hm_loss+um_loss+hm3_loss

        tf.summary.scalar('tra/um_loss', um_loss)
        tf.summary.scalar('tra/hm_loss', hm_loss)
        tf.summary.scalar('tra/hm3_loss', hm3_loss)

        # to visualize the training error, 
        # only pick the first three for tensorboard visualization
        est_hms = est_hm_list[-1][0:3,:,:,:]
        est_ums = est_um_list[-1][0:3,:,:,:]
        est_hm3s = est_hm3_list[-1][0:3,:,:,:]
        tiny_normed_dms = tiny_normed_dms[0:3,:,:,:]
        cfgs = cfgs[0:3,:]
        coms = coms[0:3,:]
        dms = dms[0:3,:,:,:]
        est_oms = self._resume_om(est_hm3s, est_ums)

        # get point estimation
        est_normed_poses = self._xyz_estimation(est_hms, est_oms, est_hm3s, tiny_normed_dms, cfgs, coms)
        est_normed_poses = tf.reshape(est_normed_poses,
                                      (est_normed_poses.get_shape()[0].value, -1))
        xyz_pts = unnorm_xyz_pose(est_normed_poses, coms)

        # 2d joint detection
        def to_uvd_fn(elem):
            xyz_pt, cfg = elem[0], elem[1]
            return [data.util.xyz2uvd_op(xyz_pt, cfg), cfg]
        uvd_pts, _ = tf.map_fn(to_uvd_fn, [xyz_pts, cfgs])
        resized_hms = tf.image.resize_images(est_hms, (self._input_height, self._input_width), 2)
        hm_uvd_pts, _ = self._uvd_estimation_op(resized_hms, tf.ones_like(resized_hms))

        # for visualization
        gt_xy_angle = self._vis_um_xy(gt_ums)
        gt_z_angle = self._vis_um_z(gt_ums)
        est_xy_angle = self._vis_um_xy(est_ums)
        est_z_angle = self._vis_um_z(est_ums)

        if FLAGS.debug_level > 0:
            tf.summary.image('tra_dm/', dms)
            tf.summary.image('tra_pts/', tf_jointplot_wrapper(tf.squeeze(dms,axis=-1), 
                                                              tf.reshape(uvd_pts, (3,-1,3))))
            tf.summary.image('tra_pt_hm/', tf_jointplot_wrapper(tf.squeeze(dms, axis=-1),
                                                                tf.reshape(hm_uvd_pts, (3,-1,3))))
        if FLAGS.debug_level > 1:
            tf.summary.image('tra_hm_est_0/', tf_heatmap_wrapper(est_hms[:,:,:,0]))
            tf.summary.image('tra_hm_gt_0/', tf_heatmap_wrapper(gt_hms[:,:,:,0]))
            tf.summary.image('tra_3d_hm_est_0/', tf_heatmap_wrapper(est_hm3s[:,:,:,0]))
            tf.summary.image('tra_3d_hm_gt_0/', tf_heatmap_wrapper(gt_hm3s[:,:,:,0]))
            tf.summary.image('tra_um_xy_gt_0', tf_heatmap_wrapper(gt_xy_angle[:,:,:,0]))
            tf.summary.image('tra_um_z_gt_0', tf_heatmap_wrapper(gt_z_angle[:,:,:,0]))
            tf.summary.image('tra_um_xy_est_0', tf_heatmap_wrapper(est_xy_angle[:,:,:,0]))
            tf.summary.image('tra_um_z_est_0', tf_heatmap_wrapper(est_z_angle[:,:,:,0]))

        if FLAGS.debug_level > 2:
            tf.summary.image('tra_hm_gt_1/', tf_heatmap_wrapper(gt_hms[:,:,:,5]))
            tf.summary.image('tra_hm_est_1/', tf_heatmap_wrapper(est_hms[:,:,:,5]))
            tf.summary.image('tra_3d_hm_est_1/', tf_heatmap_wrapper(est_hm3s[:,:,:,5]))
            tf.summary.image('tra_3d_hm_gt_1/', tf_heatmap_wrapper(gt_hm3s[:,:,:,5]))
            tf.summary.image('tra_um_xy_est_1', tf_heatmap_wrapper(est_xy_angle[:,:,:,5]))
            tf.summary.image('tra_um_z_est_1', tf_heatmap_wrapper(est_z_angle[:,:,:,5]))
            tf.summary.image('tra_um_xy_gt_1', tf_heatmap_wrapper(gt_xy_angle[:,:,:,5]))
            tf.summary.image('tra_um_z_gt_1', tf_heatmap_wrapper(gt_z_angle[:,:,:,5]))

        return total_loss 

    def opt(self, lr):
        '''return the optimizer of the model
        '''
        return tf.train.AdamOptimizer(lr, beta1=self._adam_beta1)

    # validation and test
    def test(self, dms, poses, cfgs, coms, reuse_variables=True):
        '''the validation step to show the result from the validation set

        '''
        batch_size = dms.get_shape()[0].value
        # 1st phase, gpu computation
        normed_dms = data.preprocess.norm_dm(dms, coms)
        end_points = self.inference(normed_dms, cfgs, coms, reuse_variables=reuse_variables, is_training=False)

        est_hms = end_points['hm_outs'][-1]
        
        tiny_normed_dms = tf.image.resize_images(normed_dms, (self._output_height, self._output_width), 2)
        est_ums = end_points['um_outs'][-1]
        est_hm3s = end_points['hm3_outs'][-1]

        est_oms = self._resume_om(est_hm3s, est_ums)

        est_normed_poses = self._xyz_estimation(est_hms, est_oms, est_hm3s, tiny_normed_dms, cfgs, coms)
        est_normed_poses = tf.reshape(est_normed_poses,
                                      (est_normed_poses.get_shape()[0].value, -1))
        xyz_pts = unnorm_xyz_pose(est_normed_poses, coms)

        def to_uvd_fn(elem):
            xyz_pt, cfg = elem[0], elem[1]
            return [data.util.xyz2uvd_op(xyz_pt, CameraConfig(*tf.unstack(cfg,axis=0))), cfg]
        uvd_pts, _ = tf.map_fn(to_uvd_fn, [xyz_pts, cfgs])
        gt_uvd_pts, _ = tf.map_fn(to_uvd_fn, [poses, cfgs])

        resized_hms = tf.image.resize_images(est_hms, (self._input_height, self._input_width))
        hm_uvd_pts, _ = self._uvd_estimation_op(resized_hms, tf.ones_like(resized_hms))

        # for gt visulization
        gt_normed_poses = norm_xyz_pose(poses, coms)
        gt_hms = self._hm_2d(poses, cfgs, self._output_height, self._output_width)
        xyzs = generate_xyzs_from_multi_cfgs(tiny_normed_dms, cfgs, coms)
        xyzs = tf.tile(xyzs, [1,1,1,self._jnt_num])
        gt_oms = tf.reshape(gt_normed_poses, (-1,1,1,3*self._jnt_num)) - xyzs
        gt_hm3s = self._hm_3d(gt_oms)
        gt_ums = self._um(gt_oms, gt_hm3s)
        gt_xy_angle = self._vis_um_xy(gt_ums)
        gt_z_angle = self._vis_um_z(est_ums)

        # add summayries
        est_xy_angle = self._vis_um_xy(est_ums)
        est_z_angle = self._vis_um_z(est_ums)
        if FLAGS.debug_level > 0:
            tf.summary.image('val_pts/', tf_jointplot_wrapper(tf.squeeze(dms,axis=-1), 
                                                              tf.reshape(uvd_pts, (batch_size,-1,3))),
                             collections=['val_summaries'])
            tf.summary.image('gt_pts/', tf_jointplot_wrapper(tf.squeeze(dms,axis=-1), 
                                                              tf.reshape(gt_uvd_pts, (batch_size,-1,3))),
                             collections=['val_summaries'])
        if FLAGS.debug_level > 1:
            tf.summary.image('gt_hm/', tf_heatmap_wrapper(gt_hms[:,:,:,0]), 
                             collections=['val_summaries'])
            tf.summary.image('gt_hm3', tf_heatmap_wrapper(gt_hm3s[:,:,:,0]),
                             collections=['val_summaries'])
            tf.summary.image('val_hm/', tf_heatmap_wrapper(est_hms[:,:,:,0]), 
                             collections=['val_summaries'])
            tf.summary.image('val_hm3', tf_heatmap_wrapper(est_hm3s[:,:,:,0]),
                             collections=['val_summaries'])
            tf.summary.image('val_dm/', dms, collections=['val_summaries'])
            tf.summary.image('val_pts_hm/', tf_jointplot_wrapper(tf.squeeze(dms,axis=-1), 
                                                              tf.reshape(hm_uvd_pts, (batch_size,-1,3))),
                             collections=['val_summaries'])

        if FLAGS.debug_level > 2:
            tf.summary.image('gt_xy', tf_heatmap_wrapper(gt_xy_angle[:,:,:,0]),
                             collections=['val_summaries'])
            tf.summary.image('gt_z', tf_heatmap_wrapper(gt_z_angle[:,:,:,0]),
                            collections=['val_summaries'])
            tf.summary.image('val_xy', tf_heatmap_wrapper(est_xy_angle[:,:,:,0]),
                             collections=['val_summaries'])
            tf.summary.image('val_z', tf_heatmap_wrapper(est_z_angle[:,:,:,0]),
                            collections=['val_summaries'])


        self.val_summary_op = tf.summary.merge_all(key='val_summaries')

        # interface to fetch output
        self.uvd_pts = uvd_pts
        self.xyz_pts = xyz_pts
        self.val_dms = dms
        self.est_hms = est_hms
        self.gt_pose = poses
        print('testing graph is established')

    @property
    def is_validate(self):
        return True if self._val_dataset else False

    @property
    def name(self):
        return '%s_%s'%(self._model_desc, self._net_desc)

    @property
    def train_dir(self):
        return os.path.join(self._base_dir, self.name)

    @property
    def summary_dir(self):
        return os.path.join(self.train_dir, 'summary')

    def _mean_shift(self, can_pts, num_it=10, band_width=0.8):
        '''mean shift over the candidate point
        Args:
            can_pts: candidate points, (b,j,n,3)
            num_it: number of iterations
            band_width: bandwidth of the kernel
        Returns:
            centers: the density maximal points
        '''
        def joint_fn(can_pt):
            '''iteration over joint
            Args:
                can_pt: (n,3)
            Returns:
                center: (3)
            '''
            # initialization
            num_quan = 2.0
            quan_pt = tf.clip_by_value((can_pt+1.0)*num_quan, 0, 2*num_quan-0.1)
            quan_pt = tf.to_int64(quan_pt)

            quan_hm = tf.scatter_nd(quan_pt, tf.ones(num_pt,), 
                                    (int(2*num_quan),int(2*num_quan),int(2*num_quan)))  
            curr_pt = tf.where(tf.equal(quan_hm, tf.reduce_max(quan_hm)))[-1]
            curr_pt = tf.divide(tf.to_float(curr_pt), num_quan) - 1.0
            curr_pt += 0.5/num_quan 

            # iteration
            for _ in range(num_it):
                s = tf.reduce_sum((can_pt - curr_pt)**2, axis=-1)
                s = tf.expand_dims(tf.exp(inverse_sigma*s), axis=-1)
                curr_pt = tf.reduce_sum(tf.multiply(can_pt, s), axis=0)
                curr_pt = tf.divide(curr_pt, tf.reduce_sum(s))
                curr_pt = tf.reshape(curr_pt, (1,3))

            curr_pt = tf.reshape(curr_pt, (3,))
            return curr_pt

        def batch_fn(can_pt):
            '''iteration over batch
            Args:
                can_pt: (j,n,3)
            Returns:
                centers: (j,3)
            '''
            return tf.map_fn(joint_fn ,can_pt)
        
        num_jnt = can_pts.get_shape()[1].value
        num_pt = can_pts.get_shape()[2].value
        inverse_sigma = -1.0 / (2*band_width*band_width)
        
        return tf.map_fn(batch_fn, can_pts)

    def _generate_candidates(self, hms, xyzs, num_pt):
        '''generate the candidates to do mean shift, from xyzs
        Args:
            hms: estimated heatmaps, (b,h,w,j)
            xyzs: the xyz points, (b,h,w,j*3)
            num_pt: the number of candidates
        Returns:
            can_pts: candidate points, (b,j,n,3)
        '''
        def fn(elems):
            hm, xyz = elems[0], elems[1]
            hm = tf.reshape(hm, (-1, jnt_num))
            xyz = tf.reshape(xyz, (-1, 3*jnt_num))

            hm_list = tf.unstack(hm, axis=-1)
            xyz_list = tf.unstack(xyz, axis=-1)
            can_list = []

            for j in range(jnt_num):
                weights, indices = tf.nn.top_k(hm_list[j], k=num_pt, sorted=True)
                xx = tf.gather(xyz_list[j*3], indices)
                yy = tf.gather(xyz_list[j*3+1], indices)
                zz = tf.gather(xyz_list[j*3+2], indices)
                can_list.append(tf.stack([xx,yy,zz], axis=1))
            can_pts = tf.stack(can_list, axis=0)
            return [can_pts, hms]

        jnt_num = hms.get_shape()[-1].value
        can_pts, _ = tf.map_fn(fn, [hms, xyzs])
        return can_pts

    def _get_candidate_weights(self, xyz_pts, coms, cfgs, hms, dms):
        '''the weights measures how xyz_pts fits the 2d hms estimation and dms observation
        Args:
            xyz_pts: the candidate points, (b,j,n,3)
            coms: centers of mass, (b,3)
            cfgs: camera configurations, (b,6)
            hms: estimated 2D heatmap, (b,h,w,j)
            dms: depth map, (b,h,w,1)
        Returns:
            weights: the weights of the corresponding points, (b,j,n,1)
        '''
        def fn(elems):
            xyz_pt, com, cfg, hm, dm = elems[0], elems[1], elems[2], elems[3], elems[4]

            xx,yy,zz = tf.unstack(tf.reshape(xyz_pt,(-1,3)), axis=-1)
            xyz_pt = tf.reshape(xyz_pt, (-1,))

            xyz_pt = tf.multiply(xyz_pt, data.preprocess.POSE_NORM_RATIO) + tf.tile(com,[jnt_num*pnt_num])
            xyz_pt = tf.reshape(xyz_pt, (-1,3))

            w_ratio = cfg[4] / out_w
            h_ratio = cfg[5] / out_h
            new_cfg = CameraConfig(cfg[0]/w_ratio, cfg[1]/h_ratio,
                                   cfg[2]/w_ratio, cfg[3]/h_ratio,
                                   out_w, out_h)
            uvd_pt = xyz2uvd_op(xyz_pt, new_cfg)
            uvd_pt = tf.reshape(uvd_pt, (-1, 3))
            uu, vv, dd = tf.unstack(uvd_pt, axis=-1)
            uu = tf.to_int32(uu+0.5)
            vv = tf.to_int32(vv+0.5)
            jj = tf.tile(tf.expand_dims(tf.range(jnt_num),axis=-1), [1,pnt_num])
            jj = tf.reshape(jj, (-1,))

            indices = tf.stack([vv,uu,jj], axis=-1)
            weights = tf.gather_nd(hm, indices)
            weights = tf.reshape(weights, (jnt_num, pnt_num, 1))

            #we also clip the values of depth 
            dm = tf.squeeze(dm)
            dm = tf.divide(dm*data.preprocess.D_RANGE - data.preprocess.D_RANGE*0.5, 
                           data.preprocess.POSE_NORM_RATIO)
            indices = tf.stack([vv,uu], axis=-1)
            od = tf.gather_nd(dm, indices)
            zz = tf.maximum(zz, od)
            xyz_pt = tf.stack([xx,yy,zz], axis=-1)
            xyz_pt = tf.reshape(xyz_pt, (jnt_num, pnt_num, 3))

            return [weights, xyz_pt, cfg, hm, dm]
        
        out_h, out_w = self._output_height, self._output_width
        jnt_num = xyz_pts.get_shape()[1].value
        pnt_num = xyz_pts.get_shape()[2].value
        weights, xyz_pts, _, _, _ = tf.map_fn(fn, [xyz_pts, coms, cfgs, hms, dms])
        return weights, xyz_pts

    def _weighted_mean_shift(self, can_pts, weights, num_it, band_width):
        '''mean shift over the candidate point
        Args:
            can_pts: candidate points, (b,j,n,3)
            weights: weights of candidate points, (b,j,n,1)
            num_it: number of iterations
            band_width: bandwidth of the kernel
        Returns:
            centers: the density maximal points
        '''
        def joint_fn(elems):
            '''iteration over joint
            Args:
                can_pt: (n,3), elems[0]
                weight: (n,1), elems[1]
            Returns:
                center: (3)
            '''
            can_pt, weight = elems[0], elems[1]
            # initialization
            num_quan = 2.0
            quan_pt = tf.clip_by_value((can_pt+1.0)*num_quan, 0, 2*num_quan-0.1)
            quan_pt = tf.to_int64(quan_pt)

            quan_hm = tf.scatter_nd(quan_pt, tf.squeeze(weight), 
                                    (int(2*num_quan),int(2*num_quan),int(2*num_quan)))  
            curr_pt = tf.where(tf.equal(quan_hm, tf.reduce_max(quan_hm)))[-1]
            curr_pt = tf.divide(tf.to_float(curr_pt), num_quan) - 1.0
            curr_pt += 0.5/num_quan 

            # iteration
            for _ in range(num_it):
                s = tf.reduce_sum((can_pt - curr_pt)**2, axis=-1)
                s = tf.expand_dims(tf.exp(inverse_sigma*s), axis=-1)
                s = tf.multiply(s, weight)
                curr_pt = tf.reduce_sum(tf.multiply(can_pt, s), axis=0)
                curr_pt = tf.divide(curr_pt, tf.reduce_sum(s))
                curr_pt = tf.reshape(curr_pt, (1,3))

            curr_pt = tf.reshape(curr_pt, (3,))
            return [curr_pt, can_pt]

        def batch_fn(elems):
            '''iteration over batch
            Args:
                can_pt: (j,n,3), elems[0]
                weights: (j,n,1), elems[1]
            Returns:
                centers: (j,3)
            '''
            return tf.map_fn(joint_fn ,elems)
        
        num_jnt = can_pts.get_shape()[1].value
        num_pt = can_pts.get_shape()[2].value
        inverse_sigma = -1.0 / (2*band_width*band_width)
        
        centers, _ = tf.map_fn(batch_fn, [can_pts, weights])
        return centers
    
    def _xyz_estimation(self, hms, oms, hm3s, dms, cfgs, coms):
        '''use meanshift to get the final estimation
        Args:
            hms: the heatmap returned from 2D joint detection, (b,h,w,j)
            oms: the 3D offset maps, (b,h,w,3*j)
            hm3s: the 3D heaetmap, (b,h,w,j)
            dms: the normalized depth map, (b,h,w,1)
            cfgs: camera configurations, (b,6)
        Returns:
            xyz_pts: the estimated 3d joint, (b,3*j)
        '''
        # get dense joint estimation
        jnt_num = hms.get_shape()[-1].value
        xyzs = generate_xyzs_from_multi_cfgs(dms, cfgs, coms)
        xyzs = tf.tile(xyzs, [1,1,1,self._jnt_num])
        orig_xyzs= xyzs

        xyzs = xyzs + oms

        # get the weight map for candidate selection
        # refined_hms = tf.multiply(hms, hm3s)
        refined_hms = tf.multiply(hms+1.0, hm3s)
        # refined_hms = hm3s
        # refined_hms = hms
        dms_mask = tf.where(tf.less(dms, -0.99), tf.zeros_like(dms), tf.ones_like(dms))
        refined_hms = tf.multiply(refined_hms, dms_mask)

        num_pt = 5
        can_pts = self._generate_candidates(refined_hms, xyzs, num_pt=num_pt)

        # weighted scheme
        can_weights, _ = self._get_candidate_weights(can_pts, coms, cfgs, hms, dms)
        xyz_pts = self._weighted_mean_shift(can_pts, can_weights, num_it=10, band_width=0.4)

        # unweighted scheme
        # xyz_pts = self._mean_shift(can_pts, num_it=10, band_width=0.4)

        # for visualization
        ori_pts = self._generate_candidates(refined_hms, orig_xyzs, num_pt=num_pt)

        self.can_pts = can_pts
        self.ori_pts = ori_pts 
        return xyz_pts


    def _uvd_estimation_op(self, hms, ds):
        ''' find the argmax from heatmaps and corresponding depth maps, and get the final estimation
        Args:
            hms: the heatmap with the same size as the initial captured image by camera
            ds: the depth value of the coresponding points
        Returns:
            the uvd points of the joint
        '''
        width = hms.shape[2]

        def fn(elems):
            hough_hm, hough_dm = elems[0], elems[1]
            uvd_pts = []

            hough_hm_list = tf.unstack(hough_hm, axis=-1)
            hough_dm_list = tf.unstack(hough_dm, axis=-1)
            for j in range(self._jnt_num):
                hh = hough_hm_list[j]
                hd = hough_dm_list[j]

                idx = tf.where(tf.equal(hh, tf.reduce_max(hh)))
                dd = tf.gather_nd(hd, idx)

                uu, vv, dd = tf.cast(idx[0][1],tf.float32), tf.cast(idx[0][0], tf.float32), dd[0]
                uvd_pts.append(tf.stack([uu,vv,dd]))
            return [tf.concat(uvd_pts, axis=-1), ds]
        return tf.map_fn(fn, [hms, ds])

    def do_test(self, sess, summary_writer, step, names=None):
        '''execute computation of the inference
        a fast version of inference
        '''
        # during training
        if names is None:
            f = open(self._log_path, 'a')
            summary_str, gt_vals, xyz_vals = sess.run(
                [self.val_summary_op, self.gt_pose, self.xyz_pts])
            summary_writer.add_summary(summary_str, step)

            maxJntError=[]
            f.write('[%s] step %d\n'%(datetime.now(), step))
            for xyz_val, gt_val in zip(xyz_vals, gt_vals):
                maxJntError.append(Evaluation.maxJntError(xyz_val, gt_val))
                diff = (xyz_val-gt_val).reshape(-1,3)
                dist = alg.norm(diff, axis=1).reshape(-1,1)
                error_mat = np.concatenate((diff, dist), axis=1)
                print(error_mat)
                f.write(np.array_str(error_mat)+'\n')
            print('validate error:', maxJntError)
            f.write('validation error: {}\n'.format(maxJntError))
            f.flush()
            f.close()
            return

        if step%100 == 0:
            summary_str, xyz_vals, gt_vals, names = sess.run(
                [self.val_summary_op, self.xyz_pts, self.gt_pose, names])
            summary_writer.add_summary(summary_str, step)

            maxJntError=[]
            for xyz_val, gt_val in zip(xyz_vals, gt_vals):
                maxJntError.append(Evaluation.maxJntError(xyz_val, gt_val))
                diff = (xyz_val-gt_val).reshape(-1,3)
                dist = alg.norm(diff, axis=1).reshape(-1,1)
                print(np.concatenate((diff, dist), axis=1))
            print('[step: %d]test error:'%step, maxJntError)
            print('---\n')
            return gt_vals, xyz_vals, names

        gt_vals, xyz_vals, names = sess.run([self.gt_pose, self.xyz_pts, names])
        return gt_vals, xyz_vals, names

'''unit test
'''
def run_train(dataset, val_dataset, restore_step=None):
    net_module_name = 'network.'+FLAGS.net_module

    net_module = importlib.import_module(net_module_name, package=None)
    net = net_module.detect_net
    net_name = net_module.TOWER_NAME

    model = JointDetectionModel(dataset, net, epoch=FLAGS.epoch, net_desc=net_name,
                               val_dataset = val_dataset) 
    train(model, restore_step)

def run_test(train_dataset, test_dataset, selected_step=None):
    net_module_name = 'network.'+FLAGS.net_module

    net_module = importlib.import_module(net_module_name, package=None)
    net = net_module.detect_net
    net_name = net_module.TOWER_NAME

    model = JointDetectionModel(train_dataset, net, epoch=FLAGS.epoch, net_desc=net_name,
                               val_dataset = test_dataset)

    test(model, selected_step)

if __name__ == '__main__':
    if FLAGS.dataset == 'bighand':
        import data.bigHand
        dataset = data.bigHand.BigHandDataset('training')
        val_dataset = data.bigHand.BigHandDataset('testing')

    elif FLAGS.dataset == 'nyu':
        import data.nyu
        dataset = data.nyu.NyuDataset('training')
        val_dataset = data.nyu.NyuDataset('testing')

    elif FLAGS.dataset == 'icvl':
        import data.icvl
        dataset = data.icvl.IcvlDataset('training')
        val_dataset = data.icvl.IcvlDataset('testing')

    elif FLAGS.dataset == 'msra':
        import data.msra
        dataset = data.msra.MsraDataset('training', FLAGS.pid)
        val_dataset = data.msra.MsraDataset('testing', FLAGS.pid)

    if FLAGS.is_train:
        run_train(dataset, val_dataset)
    else:
        run_test(dataset, val_dataset, -1)
