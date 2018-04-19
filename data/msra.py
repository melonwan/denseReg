from __future__ import print_function, division, absolute_import

from data.dataset_base import *
from data.dataset_base import _float_feature, _bytes_feature
import data.dataset_base
import scipy.io as sio
from data.preprocess import crop_from_xyz_pose, crop_from_bbx, center_of_mass
from data.util import uvd2xyz, xyz2uvd 
Annotation = data.dataset_base.Annotation
import cv2, struct, numpy as np

class MsraDataset(BaseDataset):
    cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
    approximate_num_per_file = 85 
    max_depth = 1000.0
    pose_dim = 63 
    jnt_num = 21
    pose_list = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split()
    
    '''nyu hand dataset contrains train, test sub-directories, both with annotated joint
    '''
    directory = './exp/data/msra15/'

    def __init__(self, subset, pid):
        if not subset in set(['training', 'testing']):
            raise ValueError('unknown sub %s set to MSRA hand datset'%subset)
        super(MsraDataset, self).__init__(subset)

        self.src_dir = os.path.join(self.directory, 'P%d'%pid)
        assert os.path.exists(self.src_dir)
        self.img_dir = self.src_dir
        self.tf_dir = os.path.join(self.directory, 'tf')
        self.pid = pid
        self.name = 'msra_P%d'%(self.pid)

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
            files = []
            for pid in range(9):
                if pid == self.pid:
                    continue
                files += [os.path.join(self.tf_dir,'P%d-%d-of-100'%(self.pid, i)) for i in range(100)]
            files += [files[-1]]
            print('[data.MsraDataset] total files for training=%d'%len(files))
            return files

        elif self.subset == 'testing':
            files = [os.path.join(self.tf_dir,'P%d-%d-of-100'%(self.pid, i)) for i in range(100)]
            files += [files[-1]]
            print('[data.MsraDataset] total files for testing=%d'%len(files))
            return files

    @property
    def approximate_num(self):
        return self.approximate_num_per_file*len(self.filenames)

    pid_num = [8499, 8492, 8412, 8488, 8500, 8497, 8497, 8498, 8492]
    @property
    def exact_num(self):
        if self.subset in set(['training', 'training_small', 'validation']):
            return self.approximate_num
        elif self.subset == 'testing':
            return self.pid_num[self.pid] 

    '''compress part: 
     to convert initial dataset TFRecord files
    '''
    def loadAnnotation(self):
        t1 = time.time()
        path = os.path.join(self.src_dir, 'labels.pkl')

        if os.path.exists(path): 
        # if False:
            with open(path, 'rb') as f:
                t1 = time.time()
                self._annotations = cPickle.load(f)
        else:
            print('[data.msra] pkl files does not exist, need to load from txt file')
            self._annotations = []
            t1 = time.time()
            for pose_name in self.pose_list:
                path = os.path.join(self.src_dir, pose_name, 'joint.txt')
                with open(path, 'r') as f:
                    for frmIdx, line in enumerate(f):
                        if frmIdx == 0:
                            continue

                        buf = line.split()
                        name = os.path.join(pose_name, '%06i_depth'%(frmIdx-1)) 
                        pose = []
                        for idx, d in enumerate(buf):
                            if idx%3 == 0:
                                pose.append(float(d))
                            elif idx%3 == 1:
                                pose.append(-float(d))
                            elif idx%3 == 2:
                                pose.append(-float(d))
                        self._annotations.append(Annotation(name, pose))

            path = os.path.join(self.src_dir, 'labels.pkl')
            with open(path, 'wb') as f:
                cPickle.dump(self._annotations, f, protocol=cPickle.HIGHEST_PROTOCOL)

        print('[data.msra]annotation has been loaded with %d samples, %fs'%\
              (len(self._annotations), time.time()-t1))

    def cvtBin2Png(self):
        MSRA_size = namedtuple('MSRA_size', ['cols', 'rows', 'left', 'top', 'right', 'bottom'])

        prevDmData = None
        self.loadAnnotation()
        for idx, anno in enumerate(self._annotations):
            path = os.path.join(self.img_dir, anno.name+'.bin')
            with open(path, 'rb') as f:
                shape = [struct.unpack('i', f.read(4))[0] for i in range(6)]
                shape = MSRA_size(*shape)
                cropDmData = np.fromfile(f, dtype=np.float32)

            crop_rows, crop_cols = shape.bottom - shape.top, shape.right - shape.left
            cropDmData = cropDmData.reshape(crop_rows, crop_cols)

            # expand the cropped dm to full-size make later process in a uniformed way
            dmData = np.zeros((shape.rows, shape.cols), np.float32)
            np.copyto(dmData[shape.top:shape.bottom, shape.left:shape.right], cropDmData)

            # for empty image, just copy the previous frame
            if dmData.sum() < 10:
                print('[warning] %s is empty'%anno.name)
                if prevDmData is not None:
                    dmData = prevDmData
            prevDmData = dmData.copy()

            path = os.path.join(self.img_dir, anno.name+'.png')
            cv2.imwrite(path, dmData.astype('uint16'))
            if idx%500 == 0:
                print('%d frames has been converted'%idx)

    def write_TFRecord_single_thread(self, thread_idx, thread_range, num_shards_per_thread):
        print('[P%d] Launching thread %d, with files from %d to %d'%(self.pid, thread_idx, thread_range[0], thread_range[1]))
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
            file_name = 'P%d-%d-of-%d'%(self.pid, file_idx, self.num_shards)
            file_path = os.path.join(self.tf_dir, file_name)
            print('[Thread %d] begin processing %d - %d images, to %s'%(
                thread_idx,shard[0],shard[1],file_path))
            t1 = time.time()
            sys.stdout.flush()
            self.saveSampleToRecord(range(shard[0], shard[1]), file_path)
            t2 = time.time()
            print('[Thread {}]end at ={}, with {}s'.format(thread_idx, datetime.now(), t2-t1))

    def convert_to_example(self, label):
        img_path = os.path.join(self.img_dir, label.name+'.png')
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
    for p in range(9):
        reader = MsraDataset('training', p)
        reader.loadAnnotation()
        reader.cvtBin2Png()
        reader.write_TFRecord_multi_thread(num_threads=20, num_shards=100)

if __name__ == '__main__':
    saveTFRecord()

