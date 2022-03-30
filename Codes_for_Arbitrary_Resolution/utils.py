import tensorflow as tf
import numpy as np
from collections import OrderedDict
import os
import glob
import cv2




class DataLoader(object):
    def __init__(self, data_folder):
        self.dir = data_folder
        self.datas = OrderedDict()
        self.setup()

    def setup(self):
        datas = glob.glob(os.path.join(self.dir, '*'))
        for data in sorted(datas):
            data_name = data.split('/')[-1]
            if data_name == 'input' or data_name == 'mask' :
                self.datas[data_name] = {}
                self.datas[data_name]['path'] = data
                self.datas[data_name]['frame'] = glob.glob(os.path.join(data, '*.jpg'))
                self.datas[data_name]['frame'].sort()
                self.datas[data_name]['length'] = len(self.datas[data_name]['frame'])

        print(self.datas.keys())

    def get_data_clips(self, index):
        batch = []
        data_info_list = list(self.datas.values())
        
        batch.append(np_load_frame(data_info_list[0]['frame'][index]))
        batch.append(np_load_frame(data_info_list[1]['frame'][index]))
       
        return np.concatenate(batch, axis=2)


def np_load_frame(filename):
    image_decoded = cv2.imread(filename)
    image_decoded = image_decoded.astype(dtype=np.float32)
    image_decoded = (image_decoded / 127.5) - 1.0
    return image_decoded



def load(saver, sess, ckpt_path):
    print(ckpt_path)
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))






