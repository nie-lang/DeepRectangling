import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2 

from PIL import Image
from model import RectanglingNetwork
from utils import load, save, DataLoader
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
    test_inputs_clips_tensor = tf.placeholder(shape=[batch_size, None, None, 3 * 3], dtype=tf.float32)
    
    test_input = test_inputs_clips_tensor[...,0:3]
    test_mask = test_inputs_clips_tensor[...,3:6]
    test_gt = test_inputs_clips_tensor[...,6:9]
    
    print('test input = {}'.format(test_input))
    print('test mask = {}'.format(test_mask))
    print('test gt = {}'.format(test_gt))



# define testing generator function 
with tf.variable_scope('generator', reuse=None):
    print('testing = {}'.format(tf.get_variable_scope().name))
    test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, test_warp_mask_final = RectanglingNetwork(test_input, test_mask)
   


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
        psnr_list = []
        ssim_list = []

        for i in range(0, length):
            input_clip = np.expand_dims(input_loader.get_data_clips(i), axis=0)
            

            mesh_primary, warp_image_primary, warp_mask_primary, mesh_final, warp_image_final, warp_mask_final  = sess.run([test_mesh_primary, test_warp_image_primary, test_warp_mask_primary, test_mesh_final, test_warp_image_final, test_warp_mask_final], feed_dict={test_inputs_clips_tensor: input_clip})
            
            
            warp_image = (warp_image_final[0]+1) * 127.5
            warp_gt = (input_clip[0,:,:,6:9]+1) * 127.5
            
            #psnr = skimage.measure.compare_psnr(input1*warp_one, warp*warp_one, 255)
            psnr = skimage.measure.compare_psnr(warp_image, warp_gt, 255)
            ssim = skimage.measure.compare_ssim(warp_image, warp_gt, data_range=255, multichannel=True)
            
            path = "../final_rectangling/" + str(i+1).zfill(5) + ".jpg"
            cv2.imwrite(path, warp_image)
            
            print('i = {} / {}, psnr = {:.6f}'.format( i+1, length, psnr))
            
            psnr_list.append(psnr)
            ssim_list.append(ssim)
            
            
        print("===================Results Analysis==================")   
        print('average psnr:', np.mean(psnr_list))
        print('average ssim:', np.mean(ssim_list))
        # as for FID, we use the CODE from https://github.com/bioinf-jku/TTUR to evaluate
        
        
        
    inference_func(snapshot_dir)

    






