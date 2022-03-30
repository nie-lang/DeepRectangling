import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2 

from PIL import Image
from model import RectanglingNetwork
from utils import load, DataLoader
import matplotlib.pyplot as plt
import skimage
import imageio


import constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

test_folder = constant.TEST_FOLDER
batch_size = constant.TEST_BATCH_SIZE



snapshot_dir = constant.SNAPSHOT_DIR + '/pretrained_model/model.ckpt-100000'
#snapshot_dir = './checkpoints/model.ckpt-100000'


batch_size = 1

# define dataset
with tf.name_scope('dataset'):
    ##########testing###############
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 2], dtype=tf.float32)
    
    test_input = test_inputs_clips_tensor[...,0:3]
    test_mask = test_inputs_clips_tensor[...,3:6]
    
    print('test input = {}'.format(test_input))
    print('test mask = {}'.format(test_mask))



# define testing generator function 
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_warp_image_final = RectanglingNetwork(test_input, test_mask)
   


config = tf.ConfigProto()
config.gpu_options.allow_growth = True      
with tf.Session(config=config) as sess:
    # dataset
    input_loader = DataLoader(test_folder)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init global successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)

    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)

    def inference_func(ckpt):
        print("============")
        print(ckpt)
        load(loader, sess, ckpt)
        print("============")
        length = len(os.listdir(test_folder+"/input"))

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)
            

            warp_image_final = sess.run([test_warp_image_final], feed_dict={test_inputs_clips_tensor: input_clip})
            
            warp_image_final = warp_image_final[0]
            warp_image = (warp_image_final[0]+1) * 127.5
            

            path = "./rectangling/" + str(i+1).zfill(5) + ".jpg"
            cv2.imwrite(path, warp_image)
            
            print('i = {} / {}'.format( i+1, length))
        
        
        
    inference_func(snapshot_dir)

    






