from __future__ import print_function, division, absolute_import
import collections
import cv2, numpy as np
import scipy.stats as st

import gpu_config
import tensorflow as tf

CameraConfig = collections.namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')

'''utilities for 2D-3D conversions
function with _op suffix returns a tf operation
'''

'''_pro: perspective transformation
   _bpro: back perspective transformation
'''
# fx, fy, cx, cy, w, h
# 0,  1,  2,  3,  4, 5
_pro = lambda pt3, cfg: [pt3[0]*cfg[0]/pt3[2]+cfg[2], pt3[1]*cfg[1]/pt3[2]+cfg[3], pt3[2]]
_bpro = lambda pt2, cfg: [(pt2[0]-cfg[2])*pt2[2]/cfg[0], (pt2[1]-cfg[3])*pt2[2]/cfg[1], pt2[2]] 

def xyz2uvd(xyz, cfg):
    '''xyz: list of xyz points
    cfg: camera configuration
    '''
    xyz = xyz.reshape((-1,3))
    # perspective projection function
    uvd = [_pro(pt3, cfg) for pt3 in xyz]
    return np.array(uvd)

def uvd2xyz(uvd, cfg):
    '''uvd: list of uvd points
    cfg: camera configuration
    '''
    uvd = uvd.reshape((-1,3))
    # backprojection
    xyz = [_bpro(pt2, cfg) for pt2 in uvd]
    return np.array(xyz)

def xyz2uvd_op(xyz_pts, cfg):
    '''xyz_pts: tensor of xyz points
       camera_cfg: constant tensor of camera configuration
    '''
    xyz_pts = tf.reshape(xyz_pts, (-1,3))
    xyz_list = tf.unstack(xyz_pts)
    uvd_list = [_pro(pt, cfg) for pt in xyz_list]
    uvd_pts = tf.stack(uvd_list)
    return tf.reshape(uvd_pts, shape=(-1,))

def uvd2xyz_op(uvd_pts, cfg):
    uvd_pts = tf.reshape(uvd_pts, (-1,3))
    uvd_list = tf.unstack(uvd_pts)
    xyz_list = [_bpro(pt, cfg) for pt in uvd_list]
    xyz_pts = tf.stack(xyz_list)
    return tf.reshape(xyz_pts, (-1,))

'''as a pre-processing step
'''
def _gaussian_kern(filter_size=10, sigma=3):
    '''
        return an np array of a Gaussian kernel
    '''
    interval = (2*sigma+1.0)/(filter_size)
    x = np.linspace(-sigma-interval/2., sigma+interval/2., filter_size+1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw/kernel_raw.sum()
    return kernel

def gaussian_filter(filter_size=10, sigma=3):
    gau_init = tf.constant(_gaussian_kern(filter_size,sigma), tf.float32)
    with tf.variable_scope('preprocess') as scope:
        try:
            gaussian_filter = tf.get_variable('gaussian_filter',
                                              initializer=gau_init, trainable=False)
            gaussian_filter = tf.reshape(gaussian_filter, (filter_size,filter_size,1,1))
        except ValueError:
            scope.reuse_variables()
            gaussian_filter = tf.get_variable('gaussian_filter',
                                              initializer=gau_init, trainable=False)
            gaussian_filter = tf.reshape(gaussian_filter, (filter_size,filter_size,1,1))
        return gaussian_filter

def heatmap_from_uvd_op(uvd_pts, cfg, gaussian_filter):
    '''we firstly construct a sparse tensor from the coordinate
    val: the value at the center of corresponding point
    '''
    with tf.name_scope('preprocess'):
        uvd_pts = tf.reshape(uvd_pts, (-1,3))
        num_pt = uvd_pts.shape[0]
        num_pt_op = tf.to_int64(num_pt)

        nn = tf.range(num_pt, dtype=tf.int64)
        nn = tf.reshape(nn, (-1,1))

        xx = uvd_pts[:,0]
        xx = tf.clip_by_value(xx, 0, cfg.w-1)
        xx = tf.to_int64(xx)
        xx = tf.reshape(xx, (-1,1))

        yy = uvd_pts[:,1]
        yy = tf.clip_by_value(yy, 0, cfg.h-1)
        yy = tf.to_int64(yy)
        yy = tf.reshape(yy, (-1,1))
        indices = tf.concat([nn,yy,xx], axis=1)

        val = 1.0
        raw_hm = tf.sparse_to_dense(sparse_indices=indices,
                                    output_shape=[num_pt_op,cfg.h,cfg.w],
                                    sparse_values=val)
        raw_hm = tf.expand_dims(raw_hm, axis=[-1])
        raw_hm = tf.cast(raw_hm, tf.float32)
        
        hm = tf.nn.conv2d(raw_hm, gaussian_filter, strides=[1,1,1,1], 
                         padding='SAME', data_format='NHWC')
        hm = tf.nn.conv2d(hm, gaussian_filter, strides=[1,1,1,1], 
                         padding='SAME', data_format='NHWC')
        hm = tf.divide(hm, tf.reduce_max(hm))
        
        # shuffle dimensions of hm
        hm_list = tf.unstack(hm, axis=0)
        hm = tf.concat(hm_list, axis=2)
        return hm 

def heatmap_from_xyz_op(xyz_pts, cfg, gaussian_filter):
    return heatmap_from_uvd_op(xyz2uvd_op(xyz_pts, cfg), cfg, gaussian_filter)


'''utilities for visualization
'''
def visHeatMap(dm, pose, ch_flag=None):
    raise NotImplementedError

def visDepthMap(dm, thresh=750, isHeatmap=True):
    dm[dm>thresh] = 0
    ratio = 255/thresh
    dm = dm*ratio
    if False:
        dm = dm/dm.max()
        dm_color = cv2.applyColorMap(dm, cv2.COLORMAP_JET)
        dm = dm_color
    else:
        dm = cv2.cvtColor(dm.astype('uint8'), cv2.COLOR_GRAY2BGR)
    return dm

def visAnnotatedDepthMap(dm, pose, cfg, thresh=750):
    dm = visDepthMap(dm, thresh)
    pose = xyz2uvd(pose,cfg)
    for pt2 in pose:
        cv2.circle(dm, (int(pt2[0]), int(pt2[1])), 3, (0,0,255), -1)
    return dm

def visAnnotatedDepthMap_uvd(dm, pose, thresh=750):
    dm = visDepthMap(dm, thresh)
    for pt2 in pose:
        cv2.circle(dm, (int(pt2[0]), int(pt2[1])), 3, (0,0,255), -1)
    return dm

'''unit test
'''
def run_heatmap_from_xyz():
    from data.bigHand import BigHandDataset

    pts = np.array([-67.4598, 5.3851, 584.7425, -55.6470, 8.8958, 587.4889, -35.5874, -54.6665, 583.3420, -54.7895, -53.8799, 577.8048, -71.0328, -51.3926, 573.4493, -88.8696, -46.2022, 569.1099, -32.8905, -20.8474, 553.7415, -18.7491, -39.3305, 532.7702, -19.8893, -56.4645, 516.0034, -35.5810, -69.2128, 545.6373, -35.5768, -78.8591, 520.6336, -35.2772, -75.8186, 501.8809, -52.5099, -66.7139, 535.8283, -51.0812, -74.7579, 509.5187, -51.7939, -78.6711, 488.8988, -72.3119, -85.2855, 549.0604, -73.1781, -108.2356, 532.5458, -69.9800, -125.8427, 521.5565, -101.7839, -74.5066, 557.4333, -110.1215, -92.7800, 549.8948, -117.0142, -109.9064, 545.4029
]) 
    pts = pts.reshape((-1,)).astype(np.float32)
    
    tf.reset_default_graph()
    xyz_pts = tf.placeholder(tf.float32,(BigHandDataset.pose_dim,))
    cfg = BigHandDataset.cfg
    heatmap_op = heatmap_from_xyz_op(xyz_pts, cfg)

    with tf.Session() as sess:
        (heatmap,) = sess.run([heatmap_op], {xyz_pts:pts})
        print('gaussian blurred')
        summap = np.zeros((BigHandDataset.cfg.h, BigHandDataset.cfg.w))
        print(heatmap.shape)
        for hm in heatmap:
            summap += hm

        summap /= summap.max()
        import matplotlib.pyplot as plt
        plt.imshow(summap, interpolation='none')
        plt.show()

if __name__ == '__main__':
    run_heatmap_from_xyz()
