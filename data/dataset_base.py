# the base class of dataset

from __future__ import print_function, division, absolute_import

import gpu_config
import tensorflow as tf

from collections import namedtuple
import time, os, cPickle, sys, threading, glob
from datetime import datetime
import time

import numpy as np
import cv2

from data.util import *
Annotation = namedtuple('Annotation', 'name,pose')

def _float_feature(value):
    if isinstance(value, np.ndarray):
        value = value
    elif not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class BaseDataset(object):
    '''provide basic utilities to convert the initial dataset to TFRecord files
       and the interface to define readdata on the graph
    '''
    def __init__(self, subset):
        '''subset: e.g., train, validation, test, or train_1 in the nyu (with view 1)
        '''
        self.subset = subset

    def loadAnnotation(self):
        '''the annotation is a sequential list of Annotation namedtuple 
        '''
        raise NotImplementedError

    @property
    def annotations(self):
        raise NotImplementedError
    
    def convert_to_example(self, label):
        '''load the image corresponding to the label.name, 
           then serialize the structure to tf.train.Example
        '''
        raise NotImplementedError

    def saveSampleToRecord(self, idx_list, tar_file_path):
        curr_list = [self.annotations[idx] for idx in idx_list]
        # if os.path.exists(tar_file_path):
            # print('%s alread written'%tar_file_path)
            # sys.stdout.flush()
            # return

        writer = tf.python_io.TFRecordWriter(tar_file_path)
        for label in curr_list:
            example = self.convert_to_example(label)
            writer.write(example.SerializeToString())
        writer.close()

    def write_TFRecord_single_thread(self, thread_idx, thread_range, num_shards_per_thread):
        print('Launching thread %d, with files from %d to %d'%(thread_idx, thread_range[0], thread_range[1]))
        sys.stdout.flush()
        spacing = np.linspace(thread_range[0], thread_range[1], num_shards_per_thread+1).astype(np.int)

        shard_range = []
        for idx in range(num_shards_per_thread):
            shard_range.append((spacing[idx], spacing[idx+1]))

        if not hasattr(self, 'num_shards'):
            '''in case of single thread
            '''
            self.num_shards = num_shards_per_thread

        for curr_shard_idx, shard in enumerate(shard_range):
            file_idx = thread_idx*num_shards_per_thread + curr_shard_idx
            file_name = '%s-%d-of-%d'%(self.subset, file_idx, self.num_shards)
            file_path = os.path.join(self.tf_dir, file_name)
            print('[Thread %d] begin processing %d - %d images, to %s'%(
                thread_idx,shard[0],shard[1],file_path))
            t1 = time.time()
            sys.stdout.flush()
            self.saveSampleToRecord(range(shard[0], shard[1]), file_path)
            t2 = time.time()
            print('[Thread {}]end at ={}, with {}s'.format(thread_idx, datetime.now(), t2-t1))
        
    def write_TFRecord_multi_thread(self, num_threads, num_shards):
        '''convert all the dataset to several file shards
        num_threads: number of threads to load and save the data
        num_shards: number of file segment on the harddisk
        '''
        if not os.path.exists(self.tf_dir):
            os.mkdir(self.tf_dir)

        assert not num_shards % num_threads, (
            'please make the num_threads commensurate with file_shards')
        self.num_shards = num_shards
        self.num_threads = num_threads
        num_shards_per_thread = int(num_shards/num_threads)

        self.loadAnnotation()
        
        spacing = np.linspace(0, len(self.annotations), num_threads+1).astype(np.int)
        thread_range = []
        for idx in range(num_threads):
            thread_range.append((spacing[idx], spacing[idx+1]))
        
        coord = tf.train.Coordinator()
        threads = []
        print('begin writing at ', datetime.now())
        sys.stdout.flush()
        for thread_idx in range(len(thread_range)):
            args = (thread_idx, 
                    thread_range[thread_idx],
                    num_shards_per_thread)
                    
            t = threading.Thread(target=self.write_TFRecord_single_thread, args=args)
            t.start()
            threads.append(t)

        # wait all thread end
        coord.join(threads)

    # interface to the batch iteration
    @property
    def filenames(self):
        if self.subset == 'testing':
            pattern = os.path.join(self.tf_dir, '%s-*'%'testing')
        else:
            pattern = os.path.join(self.tf_dir, '%s-*'%'training')
        files = glob.glob(pattern)
        print('[data.dataset_base]total file found = %d'%(len(files)))
        return files 

    @property
    def is_train(self):
        raise NotImplementedError

    @property
    def approximate_num(self):
        '''return:
            the approximate total number of training set
        '''
        raise NotImplementedError

    def get_batch_op(self, 
                     batch_size, num_readers=1, num_preprocess_threads=1, 
                     preprocess_op=None,
                     is_train=None):
        '''return the operation on tf graph of 
        iteration over the given dataset
        '''
        if is_train == None:
            is_train = self.is_train

        with tf.name_scope('batch_processing'):
            min_queue_examples = batch_size*1 

            if is_train:
                assert num_readers >1, 'during training, num_readers should be greater than 1, to shuffle the input' 
                filename_queue = tf.train.string_input_producer(
                    self.filenames, capacity=32, shuffle=True)

                example_queue = tf.RandomShuffleQueue(
                    capacity=self.approximate_num_per_file*8 + 3*batch_size,
                    min_after_dequeue=self.approximate_num_per_file*8,
                    dtypes=[tf.string])
                    
            else:
                filename_queue = tf.train.string_input_producer(
                    self.filenames, capacity=1, shuffle=False)
                example_queue = tf.FIFOQueue(
                    capacity=min_queue_examples+batch_size,
                    dtypes=[tf.string])
            
            if num_readers > 1:
                enqueue_ops = []
                for _ in range(num_readers):
                    reader = tf.TFRecordReader()
                    _, value = reader.read(filename_queue)
                    enqueue_ops.append(example_queue.enqueue([value]))

                tf.train.queue_runner.add_queue_runner(
                    tf.train.queue_runner.QueueRunner(example_queue, enqueue_ops))
                example_serialized = example_queue.dequeue()
            else:
                reader = tf.TFRecordReader()
                _, example_serialized = reader.read(filename_queue)

            results = []
            for thread_idx in range(num_preprocess_threads):
                dm, pose, name = self.parse_example(example_serialized)
                if preprocess_op != None:
                    result = preprocess_op(dm, pose, self.cfg)
                    results.append(list(result))
                else:
                    results.append([dm, pose])

            batch = tf.train.batch_join(
                results, batch_size=batch_size, capacity=2*num_preprocess_threads*batch_size)

            return batch

    # TODO: merge this function to get_batch_op
    def get_batch_op_test(self, batch_size, preprocess_op=None):
        '''return the operation on tf graph of 
        iteration over the given dataset
        '''
        with tf.name_scope('batch_processing'):
            min_queue_examples = 1 

            filename_queue = tf.train.string_input_producer(
                self.filenames, num_epochs=1, capacity=1, shuffle=False)
            example_queue = tf.FIFOQueue(
                capacity=10,
                dtypes=[tf.string])
            
            reader = tf.TFRecordReader()
            _, example_serialized = reader.read(filename_queue)

            results = []

            dm, pose, name = self.parse_example(example_serialized)
            if preprocess_op != None:
                result = preprocess_op(dm, pose, self.cfg)
                results.append(list(result)+[name])
            else:
                results.append([dm, pose, name])

            batch = tf.train.batch_join(
                results, batch_size=batch_size, capacity=2)
            return batch

    def parse_example(self, example_serialized):
        raise NotImplementedError
