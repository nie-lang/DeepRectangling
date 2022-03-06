import tensorflow as tf
import os
import time
import numpy as np
import pickle
import cv2 as cv 

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
grid_w = constant.GRID_W
grid_h = constant.GRID_H
    
def draw_mesh_on_warp(warp, f_local):
    
    #f_local[3,0,0] = f_local[3,0,0] - 2
    #f_local[4,0,0] = f_local[4,0,0] - 4
    #f_local[5,0,0] = f_local[5,0,0] - 6
    #f_local[6,0,0] = f_local[6,0,0] - 8
    #f_local[6,0,1] = f_local[6,0,1] + 7
    
    min_w = np.minimum(np.min(f_local[:,:,0]), 0).astype(np.int32)
    max_w = np.maximum(np.max(f_local[:,:,0]), 512).astype(np.int32)
    min_h = np.minimum(np.min(f_local[:,:,1]), 0).astype(np.int32)
    max_h = np.maximum(np.max(f_local[:,:,1]), 384).astype(np.int32)
    cw = max_w - min_w
    ch = max_h - min_h
    
    pic = np.ones([ch+10, cw+10, 3], np.int32)*255
    #x = warp[:,:,0]
    #y = warp[:,:,2]
    #warp[:,:,0] = y
    #warp[:,:,2] = x
    pic[0-min_h+5:0-min_h+384+5, 0-min_w+5:0-min_w+512+5, :] = warp
    
    warp = pic
    f_local[:,:,0] = f_local[:,:,0] - min_w+5
    f_local[:,:,1] = f_local[:,:,1] - min_h+5
    
    
    
    point_color = (0, 255, 0) # BGR
    thickness = 2
    lineType = 8
    #cv.circle(warp, (60, 0), 60, point_color, 0)
    #cv.circle(warp, (f_local[0,0,0], f_local[0,0,1]), 5, point_color, 0)
    num = 1
    for i in range(grid_h+1):
        for j in range(grid_w+1):
            #cv.putText(warp, str(num), (f_local[i,j,0], f_local[i,j,1]), cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            num = num + 1
            if j == grid_w and i == grid_h:
                continue
            elif j == grid_w:
                cv.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
            elif i == grid_h:
                cv.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
            else :
                cv.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i+1,j,0], f_local[i+1,j,1]), point_color, thickness, lineType)
                cv.line(warp, (f_local[i,j,0], f_local[i,j,1]), (f_local[i,j+1,0], f_local[i,j+1,1]), point_color, thickness, lineType)
              
    return warp




snapshot_dir = './checkpoints/pretrained_model/model.ckpt-100000'


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
            
            
            mesh = mesh_final[0]
            input_image = (input_clip[0,:,:,0:3]+1)/2*255
            
            input_image = draw_mesh_on_warp(input_image, mesh)
            #input_mask = draw_mesh_on_warp(np.ones([384, 512, 3], np.int32)*255, mesh)
            
            path = "../final_mesh/" + str(i+1).zfill(5) + ".jpg"
            cv.imwrite(path, input_image)
            
            #path = "../mesh_mask/" + str(i+1).zfill(5) + ".jpg"
            #cv.imwrite(path, input_mask)
            
            print('i = {} / {}'.format( i+1, length))
                
    inference_func(snapshot_dir)





