from __future__ import print_function, division, absolute_import

from data.dataset_base import *
from data.dataset_base import _float_feature, _bytes_feature
import data.dataset_base
import scipy.io as sio
import cPickle
from data.preprocess import crop_from_xyz_pose, crop_from_bbx, center_of_mass

Annotation = namedtuple('Annotation', 'name,pose,bbx')

class NyuDataset(BaseDataset):
    cfg = CameraConfig(fx=588.235, fy=587.084, cx=320, cy=240, w=640, h=480)
    approximate_num_per_file = 730 
    name = 'nyu'
    max_depth = 1500.0
    
    '''nyu hand dataset contrains train, test sub-directories, both with annotated joint
    '''
    # directory = './exp/data/nyu/'
    directory = '/srv/glusterfs/wanc/data/nyu/'

    def __init__(self, subset):
        if not subset in set(['training', 'training_small', 'validation', 'testing']):
            raise ValueError('unknown sub %s set to NYU hand datset'%subset)
        super(NyuDataset, self).__init__(subset)

        if subset in set(['training', 'training_small', 'validation']):
            self.src_dir = os.path.join(self.directory, 'dataset/train')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.tf_dir = os.path.join(self.directory, 'tf_train')

        elif subset == 'testing':
            self.src_dir = os.path.join(self.directory, 'dataset/test')
            assert os.path.exists(self.src_dir)
            self.img_dir = self.src_dir
            self.tf_dir = os.path.join(self.directory, 'tf_test')

        l = [0,3,6,9,12,15,18,21,24,25,27,30,31,32]
        keep_pose_idx = [[ll*3, ll*3+1, ll*3+2] for ll in l]
        self.keep_pose_idx = np.array([idx for sub_idx in keep_pose_idx for idx in sub_idx]).reshape((-1,1))
        self.orig_pose_dim = 108
        self.pose_dim = len(self.keep_pose_idx)
        self.jnt_num = int(self.pose_dim/3)
        print('[NyuDataset] we only keep %d joints, with %d dim'%(self.jnt_num, self.pose_dim))

    @property
    def annotations(self):
        return self._annotations

    @property
    def is_train(self):
        return True
        if self.subset == 'training' or self.subset == 'training_small':
            return True
        else:
            return False

    @property
    def filenames(self):
        if self.subset == 'training':
            files = [os.path.join(self.tf_dir,'training-%d-of-300'%i) for i in range(100)]
            files += [files[-1]]
            print('[data.NyuDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'training_small':
            files = [os.path.join(self.tf_dir,'training-%d-of-300'%i) for i in range(30)]
            files = [f for idx, f in enumerate(files) if idx%10==0]
            print('[data.NyuDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'validation':
            files = [os.path.join(self.tf_dir,'training-%d-of-300'%i) for i in range(100)]
            files = [f for idx, f in enumerate(files) if idx%21==0]
            print('[data.NyuDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'testing':
            files = [os.path.join(self.tf_dir,'testing-%d-of-16'%i) for i in range(16)]
            files += [files[-1]]
            print('[data.NyuDataset] total files for testing=%d'%len(files))
            return files

    @property
    def approximate_num(self):
        return self.approximate_num_per_file*len(self.filenames)

    @property
    def exact_num(self):
        if self.subset in set(['training', 'training_small', 'validation']):
            return self.approximate_num
        elif self.subset == 'testing':
            return 8252

    '''compress part: 
     to convert initial dataset TFRecord files
    '''
    def loadAnnotation(self, is_trun=False):
        '''is_trun: 
            True: to load 14 joints from self.keep_list
            False: to load all joints
        '''
        t1 = time.time()
        path = os.path.join(self.src_dir, 'joint_data.mat')
        mat = sio.loadmat(path)
        camera_num = 1 if self.subset=='testing' else 3
        joints = [mat['joint_xyz'][idx] for idx in range(camera_num)]
        names = [['depth_{}_{:07d}.png'.format(camera_idx+1, idx+1) for idx in range(len(joints[camera_idx]))] for camera_idx in range(camera_num)]

        if self.subset == 'testing':
            with open('data/nyu_bbx.pkl', 'rb') as f:
                bbxes = [cPickle.load(f)]

        self._annotations = []
        if self.subset == 'testing':
            for c_j, c_n, c_b in zip(joints, names, bbxes):
                for j, n, b in zip(c_j, c_n, c_b):
                    j = j.reshape((-1,3))
                    j[:,1] *= -1.0
                    j = j.reshape((-1,))
                    if is_trun:
                        j = j[self.keep_pose_idx]
                    b = np.asarray(b).reshape((-1,))
                    self._annotations.append(Annotation(n, j.reshape((-1,)), b))
        else:
            for c_j, c_n in zip(joints, names):
                for j, n in zip(c_j, c_n):
                    j = j.reshape((-1,3))
                    j[:,1] *= -1.0
                    j = j.reshape((-1,))
                    if is_trun:
                        j = j[self.keep_pose_idx]
                    self._annotations.append(Annotation(n, j.reshape((-1,)), None))

        print('[data.NyuDataset]annotation has been loaded with %d samples, %fs'%\
              (len(self._annotations), time.time()-t1))

    def loadImage(self, idxes):
        ''' directly load images and annotations from the source directory
        '''
        dms, poses = [], []
        for idx in idxes:
            ann = self._annotations[idx]
            img_path = os.path.join(self.img_dir, ann.name)
            dms.append(cv2.imread(img_path,-1))
            poses.append(ann.pose)
        return dms, poses

    def _decode_png(self, img_data):
        image = tf.image.decode_png(img_data, channels=3, dtype=tf.uint8)
        image = tf.reshape(image, (self.cfg.h, self.cfg.w, 3))

        _,g,b = tf.unstack(image, axis=-1)
        g,b = tf.cast(g, tf.uint16), tf.cast(b, tf.uint16)
        g = tf.multiply(g, 256) # left shift with 8 bits
        d = tf.expand_dims(tf.bitwise.bitwise_or(g, b), -1)
        return tf.to_float(d)

    def convert_to_example(self, label):
        img_path = os.path.join(self.img_dir, label.name)
        with tf.gfile.FastGFile(img_path, 'r') as f:
            img_data = f.read()
            self._decode_png(img_data)

        if self.subset == 'testing':
            example = tf.train.Example(features=tf.train.Features(feature={
                'name':_bytes_feature(label.name),
                'xyz_pose':_float_feature(label.pose),
                'bbx':_float_feature(label.bbx),
                'png16':_bytes_feature(img_data)}))
        else:
            example = tf.train.Example(features=tf.train.Features(feature={
                'name':_bytes_feature(label.name),
                'xyz_pose':_float_feature(label.pose),
                'png16':_bytes_feature(img_data)}))
        return example

    # decoding part for batch iteration interface
    def parse_example(self, example_serialized):
        feature_map = {
            'name': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'xyz_pose': tf.FixedLenFeature([self.orig_pose_dim], dtype=tf.float32),
            'png16': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        features = tf.parse_single_example(example_serialized, feature_map)
        image = self._decode_png(features['png16'])
        pose = features['xyz_pose']
        pose = tf.gather_nd(pose, self.keep_pose_idx)
        image = tf.reshape(image, [self.cfg.h,self.cfg.w,1])
        name = features['name']
        return image, pose, name 

    def parse_example_test(self, example_serialized):
        feature_map = {
            'name': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'xyz_pose': tf.FixedLenFeature([self.orig_pose_dim], dtype=tf.float32),
            'bbx': tf.FixedLenFeature([5], dtype=tf.float32),
            'png16': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        features = tf.parse_single_example(example_serialized, feature_map)
        image = self._decode_png(features['png16'])
        pose = features['xyz_pose']
        pose = tf.gather_nd(pose, self.keep_pose_idx)
        bbx = features['bbx']
        image = tf.reshape(image, [self.cfg.h,self.cfg.w,1])
        name = features['name']
        return image, pose, bbx, name 

    def preprocess_op(self, input_width, input_height):
        if self.subset == 'testing':
            def preprocess_op(dm, pose, bbx, cfg):
                dm, pose, cfg = crop_from_bbx(dm, pose, bbx, cfg, input_width, input_height)
                com = center_of_mass(dm, cfg) 
                return [dm, pose, cfg, com]
            return preprocess_op
        else: 
            def preprocess_op(dm, pose, cfg):
                dm, pose, cfg = crop_from_xyz_pose(dm, pose, cfg, input_width, input_height)
                com = center_of_mass(dm, cfg) 
                return [dm, pose, cfg, com]
            return preprocess_op

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

            dm, pose, bbx, name = self.parse_example_test(example_serialized)
            if preprocess_op != None:
                result = preprocess_op(dm, pose, bbx, self.cfg)
                results.append(list(result)+[name])
            else:
                results.append([dm, pose, name])

            batch = tf.train.batch_join(
                results, batch_size=batch_size, capacity=2)
            return batch

    def get_batch_op(self, 
                     batch_size, num_readers=1, num_preprocess_threads=1, 
                     preprocess_op=None,
                     is_train=None):
        '''return the operation on tf graph of 
        iteration over the given dataset
        '''
        if self.subset == 'testing': 
            with tf.name_scope('batch_processing'):
                min_queue_examples = batch_size*1 

                filename_queue = tf.train.string_input_producer(
                    self.filenames, capacity=32, shuffle=True)

                example_queue = tf.RandomShuffleQueue(
                    capacity=self.approximate_num_per_file*8 + 3*batch_size,
                    min_after_dequeue=self.approximate_num_per_file*8,
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
                    dm, pose, bbx, name = self.parse_example_test(example_serialized)
                    if preprocess_op != None:
                        result = preprocess_op(dm, pose, bbx, self.cfg)
                        results.append(list(result))
                    else:
                        results.append([dm, pose])
                batch = tf.train.batch_join(
                    results, batch_size=batch_size, capacity=2*num_preprocess_threads*batch_size)
                return batch
        else:
            return super(NyuDataset, self).get_batch_op(batch_size, 
                                                        num_readers, 
                                                        num_preprocess_threads,
                                                        preprocess_op,
                                                        is_train)

def saveTFRecord():
    # reader = NyuDataset('training')
    # reader.write_TFRecord_multi_thread(num_threads=30, num_shards=300)

    reader = NyuDataset('testing')
    reader.write_TFRecord_multi_thread(num_threads=16, num_shards=16)

if __name__ == '__main__':
    saveTFRecord()
