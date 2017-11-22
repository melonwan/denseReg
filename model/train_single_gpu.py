from __future__ import print_function, absolute_import, division

import gpu_config
import tensorflow as tf
import network.slim as slim
import numpy as np
import time, os
from datetime import datetime

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

def train(model, restore_step=None):
    '''train the provided model
    model: provide several required interface to train
    '''
    with tf.Graph().as_default():
        global_step = tf.get_variable(
            'global_step', [],
            initializer=tf.constant_initializer(0), trainable=False)
        lr = tf.train.exponential_decay(model.init_lr,
                                       global_step,
                                       model.decay_steps,
                                       model.lr_decay_factor,
                                       staircase=True)

        print('[train] learning rate decays per %d steps with rate=%f'%(
            model.decay_steps,model.lr_decay_factor))
        print('[train] initial learning_rate = %f'%model.init_lr)
        tf.summary.scalar('learning_rate', lr)
        opt = model.opt(lr)
        
        batches = model.batch_input(model.train_dataset)

        loss = model.loss(*batches)
        tf.summary.scalar('loss', loss)

        if model.is_validate:
            # set batch_size as 3 since tensorboard visualization
            val_batches = model.batch_input(model.val_dataset, 3)
            model.test(*val_batches) # don't need the name

        batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)

        accu_steps = float(FLAGS.sub_batch)

        grads = opt.compute_gradients(loss)
        accum_grads = []
        for grad, var in grads:
            if grad is not None:
                accum_grads.append(tf.Variable(tf.zeros_like(grad), trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    name=var.op.name+'_accu_grad'))
            else:
                accum_grads.append(tf.Variable(tf.zeros_like(var), trainable=False,
                                    collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                    name=var.op.name+'_accu_grad'))

        reset_op = [grad.assign(tf.zeros_like(grad)) for grad in accum_grads]
        accum_op = [accum_grads[i].assign_add(grad[0]) for i, grad in enumerate(grads)if grad[0] is not None]

        ave_grad = [(tf.clip_by_value(tf.divide(accum_grads[i], accu_steps), -0.2, 0.2),
                     grad[1]) for i, grad in enumerate(grads)]
        apply_gradient_op = opt.apply_gradients(ave_grad, 
                                                global_step=global_step)

        for ave_grad, grad_and_var in zip(ave_grad, grads):
            grad, var = grad_and_var[0], grad_and_var[1]
            if grad is not None:
                tf.summary.histogram(var.op.name, var)
                tf.summary.histogram(var.op.name+'/gradients', ave_grad)

        # variable_averages = tf.train.ExponentialMovingAverage(
            # model.moving_average_decay, global_step)
        # variables_to_average = tf.trainable_variables()
        # var_1, var_2 = tf.moving_average_variables()[0], tf.moving_average_variables()[1]
        # variable_averages_op = variable_averages.apply(variables_to_average)

        batchnorm_update_op = tf.group(*batchnorm_updates)
        # group all training operations into one 
        # train_op = tf.group(apply_gradient_op, variable_averages_op)
        train_op = tf.group(apply_gradient_op)
        
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge_all()

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        sess = tf.Session(config=tf.ConfigProto(
            allow_soft_placement=True,
            log_device_placement=False))

        sess.run(init_op)
        start_step = 0
        # to resume the training
        if restore_step is not None and restore_step>0:
            checkpoint_path = os.path.join(model.train_dir, 'model.ckpt-%d'%restore_step)
            saver.restore(sess, checkpoint_path)
            start_step = restore_step

        tf.train.start_queue_runners(sess=sess)

        #TODO: change to tf.train.SummaryWriter()
        summary_writer = tf.summary.FileWriter(
            model.summary_dir,
            graph=sess.graph)

        # finally into the training loop
        print('finally into the long long training loop')

        log_path = os.path.join(model.train_dir, 'training_log.txt')
        f = open(log_path, 'a')

        for step in range(start_step, model.max_steps):
            if f.closed:
                f = open(log_path, 'a')

            start_time = time.time()
            ave_loss = 0
            sess.run(reset_op)
            for sub_step in range(int(accu_steps)):
                _, _, loss_value = sess.run([accum_op, batchnorm_update_op, loss])
                assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                ave_loss += loss_value

            _ = sess.run([train_op])
            ave_loss /= accu_steps
            duration = time.time() - start_time

            if step%5 == 0:
                format_str = '[model/train_multi_gpu] %s: step %d/%d, loss = %.3f, %.3f sec/batch, %.3f sec/sample'
                print(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps)))
                f.write(format_str%(datetime.now(), step, model.max_steps, ave_loss, duration, duration/(FLAGS.batch_size*accu_steps))+'\n')
                f.flush()

            if step%20 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)


            if step%40 == 0 and hasattr(model, 'do_test'):
                model.do_test(sess, summary_writer, step)

            if step%100 == 0 or (step+1) == model.max_steps:
                if not os.path.exists(model.train_dir):
                    os.makedirs(model.train_dir)
                checkpoint_path = os.path.join(model.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)
                print('model has been saved to %s\n'%checkpoint_path)
                f.write('model has been saved to %s\n'%checkpoint_path)
                f.flush()

        print('finish train')
        f.close()

