from __future__ import print_function, absolute_import, division

import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
import cv2
from datetime import datetime
from data.evaluation import Evaluation

FLAGS = tf.app.flags.FLAGS

def test(model, selected_step):
    with tf.Graph().as_default():
        total_test_num = model.val_dataset.exact_num

        dms, poses, cfgs, coms, names = model.batch_input_test(model.val_dataset)
        model.test(dms, poses, cfgs, coms, reuse_variables=None)

        # dms, poses, names = model.batch_input_test(model.val_dataset)
        # model.test(dms, poses, reuse_variables=None)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op)

        if selected_step is not None:
            checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%selected_step)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, checkpoint_path)
            print('[test_model]model has been resotored from %s'%checkpoint_path)

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(
            model.summary_dir+'_'+model.val_dataset.subset,
            graph=sess.graph)
        
        res_path = os.path.join(model.train_dir, '%s-%s-result'%(model.val_dataset.subset, datetime.now()))
        res_path = res_path.replace(' ', '_')

        res_txt_path = res_path+'.txt'
        if os.path.exists(res_txt_path):
            os.remove(res_txt_path)
        err_path = res_path+'_error.txt'
        f = open(res_txt_path, 'w')

        # res_vid_path = res_path+'.avi'
        # codec = cv2.cv.CV_FOURCC('X','V','I','D')
        # the output size is defined by the visualization tool of matplotlib
        # vid = cv2.VideoWriter(res_vid_path, codec, 25, (640, 480))
        
        print('[test_model]begin test')
        test_num = 0
        step = 0
        maxJntError = []
        while True:
            start_time = time.time()
            try:
                gt_vals, xyz_vals, name_vals = model.do_test(sess, summary_writer, step, names)
            except tf.errors.OutOfRangeError:
                print('run out of range')
                break

            duration = time.time()-start_time
            
            for xyz_val, gt_val, name_val in zip(xyz_vals, gt_vals, name_vals):
                maxJntError.append(Evaluation.maxJntError(xyz_val, gt_val))

                xyz_val = xyz_val.tolist()
                res_str = '%s\t%s\n'%(name_val, '\t'.join(format(pt, '.4f') for pt in xyz_val))
                res_str = res_str.replace('/', '\\')
                f.write(res_str)
                # vid.write(vis_val)
                test_num += 1
                if test_num >= total_test_num:
                    print('finish test')
                    f.close()
                    Evaluation.plotError(maxJntError, err_path)
                    return
            f.flush()
            
            if step%101 == 0:
                print('[%s]: %d/%d computed, with %.2fs'%(datetime.now(), step, model.max_steps, duration))

            step += 1


        print('finish test')
        f.close()
        Evaluation.plotError(maxJntError, 'result.txt')
