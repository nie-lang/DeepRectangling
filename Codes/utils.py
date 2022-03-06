import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2



rng = np.random.RandomState(2017)



class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup()

    def __call__(self, batch_size):
        data_info_list = list(self.datas.values())
        length = data_info_list[0]['length']

        def data_clip_generator():
            while True:
                data_clip = []
                frame_id = rng.randint(0, length-1)
                #######inputs
                data_clip.append(np_load_frame(data_info_list[1]['frame'][frame_id], 384, 512))
                data_clip.append(np_load_frame(data_info_list[2]['frame'][frame_id], 384, 512))
                data_clip.append(np_load_frame(data_info_list[0]['frame'][frame_id], 384, 512))
                data_clip = np.concatenate(data_clip, axis=2)
                
                yield data_clip


        dataset = tf.data.Dataset.from_generator(generator=data_clip_generator, output_types=tf.float32, output_shapes=[384, 512, 9])
        print('generator dataset, {}'.format(dataset))
        dataset = dataset.prefetch(buffer_size=128)
        dataset = dataset.shuffle(buffer_size=128).batch(batch_size)
        print('epoch dataset, {}'.format(dataset))

        return dataset

    def __getitem__(self, data_name):
        assert data_name in self.datas.keys(), 'data = {} is not in {}!'.format(data_name, self.datas.keys())
        return self.datas[data_name]

    def setup(self):
        datas = glob.glob(os.path.join(self.dir, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'gt' or data_name == 'input' or data_name == 'mask' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['frame'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['frame'].sort()
                self.datas[data_name]['length'] = len(self.datas[data_name]['frame'])

        print(self.datas.keys())

    def get_data_clips(self, index):
        batch = []
        data_info_list = list(self.datas.values())
        
        batch.append(np_load_frame(data_info_list[1]['frame'][index], 384, 512))
        batch.append(np_load_frame(data_info_list[2]['frame'][index], 384, 512))
        batch.append(np_load_frame(data_info_list[0]['frame'][index], 384, 512))
       
        return np.concatenate(batch, axis=2)


def np_load_frame(filename, resize_height, resize_width):
    image_decoded = cv2.imread(filename)
    if resize_height != None:
      image_resized = cv2.resize(image_decoded, (resize_width, resize_height))
    else:
      image_resized = image_decoded
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = (image_resized / 127.5) - 1.0
    return image_resized



def load(saver, sess, ckpt_path):
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')




