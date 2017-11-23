'''provide a multi-thread training scheme
WARNING: this file is still under development, is not guaranteed to work.
'''
from __future__ import print_function, absolute_import, division

import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
from datetime import datetime
import model.memory_util as memory_util

FLAGS = tf.app.flags.FLAGS

def _average_gradients(tower_grads):
    '''calcualte the average gradient for each shared variable across all towers on multi gpus
    Args:
        tower_grads: list of lists of (gradient, variable) tuples. len(tower_grads)=#tower, len(tower_grads[0])=#vars 
    Returns:
        List of paris (gradient, variable) where the gradients has been averaged across
        all towers
    '''
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # over different variables
        grads = []
        for g, _ in grad_and_vars:
            # over different towers
            expanded_g = tf.expand_dims(g,0)
            grads.append(expanded_g)

        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(model):
    '''train the provided model
    model: provide several required interface to train
    '''
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.train.exponential_decay(model.init_lr,
                                       global_step,
                                       model.decay_steps,
                                       model.lr_decay_factor,
                                       staircase=True)
        opt = model.opt(lr)
        
        '''split the batch into num_gpus groups, 
        do the backpropagation on each gpu seperately,
        then average the gradidents on each of which and update
        '''
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, (
            'the batch_size should be divisible wrt num_gpus')
        dms, poses = model.batch_input
        dm_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=dms)
        pose_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=poses)

        # calculate the gradients for each gpu
        tower_grads = []
        reuse_variables = None

        for i in range(FLAGS.num_gpus):
            # i = 1
            # with tf.device('/gpu:%d'%gpu_config.gpu_list[i]):
            with tf.device('gpu:%d'%i):
                with tf.name_scope('%s_%d'%(model.name, i)) as scope:
                    with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                    # with slim.arg_scope([slim.variables.variable], device='/gpu:%d'%gpu_config.gpu_list[i]):
                        loss = model.loss(dm_splits[i], pose_splits[i], reuse_variables)

                    # tf.get_variable_scope().reuse_variables()
                    # reuse variables after the first tower
                    reuse_variables = True
                    # only retain the summaries for the last tower
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    # retain the batch-norm optimization only from the last tower
                    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION,
                                                         scope)

                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    print('setup %dth gpu on %d'%(i, gpu_config.gpu_list[i]))

        grads = _average_gradients(tower_grads)

        # TODO: add input summaries
        # summaries.extend(input_summaries)

        summaries.append(tf.summary.scalar('learning_rate', lr))

        for grad, var in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name+'/gradients', grad))

        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))

        variable_averages = tf.train.ExponentialMovingAverage(
            model.moving_average_decay, global_step)
        variables_to_average = (tf.trainable_variables()+
                                tf.moving_average_variables())
        variable_averages_op = variable_averages.apply(variables_to_average)

        batchnorm_update_op = tf.group(*batchnorm_updates)
        # group all training operations into one 
        train_op = tf.group(apply_gradient_op, variable_averages_op, batchnorm_update_op)

        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        init_op = tf.global_variables_initializer()

        memory_util.vlog(1)

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        sess.run(init_op)
        tf.train.start_queue_runners(sess=sess)

        summary_writer = tf.summary.FileWriter(
            model.train_dir,
            graph=sess.graph)
        
        # finally into the training loop
        print('finally into the long long training loop')

        # for step in range(model.max_steps):
        for step in range(1000):
            start_time = time.time()
            _, loss_value = sess.run([train_op, loss])
            duration = time.time() - start_time

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            if step%10 == 0:
                format_str = '[model/train_multi_gpu] %s: step %d, loss = %.2f, %.3f sec/batch, %.3f sec/sample'
                print(format_str %(datetime.now(), step, loss_value, duration, duration/FLAGS.batch_size))

            if step%100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)

            if step%1000 == 0 or (step+1) == model.max_steps:
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

        print('finish train')

