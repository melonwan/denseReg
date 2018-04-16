from __future__ import print_function, division, absolute_import

from data.dataset_base import *
from data.dataset_base import _float_feature, _bytes_feature
import data.dataset_base
import scipy.io as sio
from data.preprocess import crop_from_xyz_pose, crop_from_bbx, center_of_mass
from data.util import uvd2xyz, xyz2uvd 
Annotation = data.dataset_base.Annotation

class IcvlDataset(BaseDataset):
    cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
    approximate_num_per_file = 220 
    name = 'icvl'
    max_depth = 500.0
    pose_dim = 48
    jnt_num = 16
    
    '''nyu hand dataset contrains train, test sub-directories, both with annotated joint
    '''
    directory = './exp/data/icvl/'

    def __init__(self, subset):
        if not subset in set(['training', 'training_small', 'validation', 'testing']):
            raise ValueError('unknown sub %s set to ICVL hand datset'%subset)
        super(IcvlDataset, self).__init__(subset)

        if subset in set(['training', 'training_small', 'validation']):
            self.src_dir = os.path.join(self.directory, 'Training')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.tf_dir = os.path.join(self.directory, 'tf_train')

        elif subset == 'testing':
            self.src_dir = os.path.join(self.directory, 'Testing')
            assert os.path.exists(self.src_dir)
            self.img_dir = os.path.join(self.src_dir, 'Depth')
            self.tf_dir = os.path.join(self.directory, 'tf_test')


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
            files = [os.path.join(self.tf_dir,'training-%d-of-100'%i) for i in range(100)]
            files += [files[-1]]
            print('[data.IcvlDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'training_small':
            files = [os.path.join(self.tf_dir,'training-%d-of-100'%i) for i in range(10)]
            files = [f for idx, f in enumerate(files) if idx%10==0]
            print('[data.IcvlDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'validation':
            files = [os.path.join(self.tf_dir,'training-%d-of-100'%i) for i in range(10)]
            files = [f for idx, f in enumerate(files) if idx%21==0]
            print('[data.IcvlDataset] total files for training=%d'%len(files))
            return files
        elif self.subset == 'testing':
            files = [os.path.join(self.tf_dir,'testing-%d-of-4'%i) for i in range(4)]
            files += [files[-1]]
            print('[data.IcvlDataset] total files for testing=%d'%len(files))
            return files

    @property
    def approximate_num(self):
        return self.approximate_num_per_file*len(self.filenames)

    @property
    def exact_num(self):
        if self.subset in set(['training', 'training_small', 'validation']):
            return self.approximate_num
        elif self.subset == 'testing':
            return 1596 

    '''compress part: 
     to convert initial dataset TFRecord files
    '''
    def loadAnnotation(self):
        t1 = time.time()
        path = os.path.join(self.src_dir, 'labels')

        if os.path.exists(path+'.pkl'): 
            with open(path+'.pkl', 'rb') as f:
                t1 = time.time()
                self._annotations = cPickle.load(f)
        else:
            print('[data.icvl] pkl files does not exist, need to load from txt file')
            with open(path+'.txt', 'r') as f:
                t1 = time.time()
                self._annotations = []
                for line in f:
                    if self.is_train and not line.startswith('2014'):
                        continue
                    buf = line.split()
                    name = buf[0]
                    pose = np.array([float(d) for d in buf[1:]])
                    pose = np.reshape(uvd2xyz(pose, self.cfg), (-1,)).tolist()
                    self._annotations.append(Annotation(name, pose))

                path = os.path.join(self.src_dir, 'labels.pkl')
                with open(path, 'wb') as f:
                    cPickle.dump(self._annotations, f, protocol=cPickle.HIGHEST_PROTOCOL)

        print('[data.icvl]annotation has been loaded with %d samples, %fs'%\
              (len(self._annotations), time.time()-t1))

    def convert_to_example(self, label):
        img_path = os.path.join(self.img_dir, label.name)
        with tf.gfile.FastGFile(img_path, 'r') as f:
            img_data = f.read()

        example = tf.train.Example(features=tf.train.Features(feature={
            'name':_bytes_feature(label.name),
            'xyz_pose':_float_feature(label.pose),
            'png16':_bytes_feature(img_data)}))
        return example

    # decoding part for batch iteration interface
    def parse_example(self, example_serialized):
        feature_map = {
            'name': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
            'xyz_pose': tf.FixedLenFeature([self.pose_dim], dtype=tf.float32),
            'png16': tf.FixedLenFeature([], dtype=tf.string, default_value=''),
        }
        features = tf.parse_single_example(example_serialized, feature_map)
        image = tf.image.decode_png(features['png16'], channels=1, dtype=tf.uint16)
        image = tf.to_float(image)
        pose = features['xyz_pose']
        image = tf.reshape(image, [self.cfg.h,self.cfg.w,1])
        name = features['name']
        return image, pose, name 

    def preprocess_op(self, input_width, input_height):
        def preprocess_op(dm, pose, cfg):
            dm, pose, cfg = crop_from_xyz_pose(dm, pose, cfg, input_width, input_height)
            com = center_of_mass(dm, cfg) 
            return [dm, pose, cfg, com]
        return preprocess_op

def saveTFRecord():
    # reader = IcvlDataset('training')
    # reader.write_TFRecord_multi_thread(num_threads=20, num_shards=100)

    reader = IcvlDataset('testing')
    reader.write_TFRecord_multi_thread(num_threads=2, num_shards=4)

def run_check_record():
    import data.util
    import matplotlib.pyplot as plt

    dataset = IcvlDataset('training')

    dms, poses = dataset.get_batch_op(batch_size=10,
                                     num_readers=4,
                                     num_preprocess_threads=1,
                                     preprocess_op=None)
    cfg = dataset.cfg
    def fn(poses):
        return data.util.xyz2uvd_op(poses, cfg)
    uvd_poses = tf.map_fn(fn, poses) 
    dms = tf.squeeze(dms, axis=-1)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    print('computational graph established')

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        dm_val, uvd_val = sess.run([dms, uvd_poses])
    coord.request_stop()
    coord.join(threads)
    print('ok with tf part')

    print(uvd_val.shape)
    print(dm_val.shape)

    for p, d in zip(uvd_val, dm_val):
        plt.clf()
        plt.imshow(d)
        p = p.reshape((-1,3))
        plt.scatter(p[:,0], p[:,1], c='r')
        plt.show()

def run_preprocess():
    import data.util
    import matplotlib.pyplot as plt

    dataset = IcvlDataset('testing')

    dms, poses, cfgs, coms = dataset.get_batch_op(
        batch_size=10,
        num_readers = 2,
        num_preprocess_threads = 2,
        preprocess_op=dataset.preprocess_op(128, 128))
    orig_dm = tf.squeeze(dms, axis=-1)
    orig_pose = poses
    dms, poses = data.preprocess.data_aug(dms, poses, cfgs, coms)

    normed_dms = data.preprocess.norm_dm(dms, coms)
    normed_dms = tf.squeeze(normed_dms, axis=-1)
    norm_poses = data.preprocess.norm_xyz_pose(poses, coms, None)

    def fn(elems):
        return [data.util.xyz2uvd_op(elems[0], CameraConfig(*tf.unstack(elems[1],axis=0))), elems[0]]
    uvd_poses,_ = tf.map_fn(fn, [poses, cfgs])

    hms, dcs = data.preprocess.pose_sync(norm_poses, normed_dms, cfgs, coms, 128, 128)
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    print('computational graph established')

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        [normed_dms_val, orig_dm_val, coms_val, pose_val, orig_pose_val, dc_val] = sess.run(
            [normed_dms, orig_dm, coms, uvd_poses, orig_pose, dcs])

    coord.request_stop()
    coord.join(threads)
    print('ok with tf part')

    for d, od, c, p, op, dc in zip(normed_dms_val, orig_dm_val, coms_val, pose_val, orig_pose_val, dc_val):
        print('c_z', c)
        print('op', op)
        print('rp', p)
        print('---')

        fig = plt.figure()
        ax = fig.add_subplot(131)
        ax.imshow(dc[:,:,0])

        ax = fig.add_subplot(132)
        ax.set_title('unexplained_map')
        ax.imshow(dc[:,:,-1])

        ax = fig.add_subplot(133)
        ax.set_title('dm')
        ax.imshow(d)
        # p = p.reshape((-1,3))
        # plt.scatter(p[:,0], p[:,1], c='r')
        plt.show()



if __name__ == '__main__':
    # saveTFRecord()
    # run_check_record()
    run_preprocess()
