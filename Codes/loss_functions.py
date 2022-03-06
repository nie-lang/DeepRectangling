import tensorflow as tf
import numpy as np

import constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H

min_w = (512/grid_w)/8
min_h = (384/grid_h)/8


# pixel-level loss (l_num=1 for L1 loss, l_num=2 for L2 loss, ......)
def intensity_loss(gen_frames, gt_frames, l_num):
    return tf.reduce_mean(tf.abs((gen_frames - gt_frames) ** l_num))


# intra-grid constraint
def intra_grid_loss(pts):
    with tf.name_scope('soft_mesh_loss2'):
        batch_size = tf.shape(pts)[0]
        
        delta_x = pts[:,:,0:grid_w,0] - pts[:,:,1:grid_w+1,0]
        delta_y = pts[:,0:grid_h,:,1] - pts[:,1:grid_h+1,:,1]
        
        loss_x = tf.nn.relu(delta_x+min_w)
        loss_y = tf.nn.relu(delta_y+min_h)
        
        loss = tf.reduce_mean(loss_x) + tf.reduce_mean(loss_y)
        
    return loss

# inter-grid constraint
def inter_grid_loss(train_mesh):
    w_edges = train_mesh[:,:,0:grid_w,:] - train_mesh[:,:,1:grid_w+1,:]
    cos_w = tf.reduce_sum(w_edges[:,:,0:grid_w-1,:] * w_edges[:,:,1:grid_w,:],3) / (tf.sqrt(tf.reduce_sum(w_edges[:,:,0:grid_w-1,:]*w_edges[:,:,0:grid_w-1,:],3))*tf.sqrt(tf.reduce_sum(w_edges[:,:,1:grid_w,:]*w_edges[:,:,1:grid_w,:],3)))
    print("cos_w.shape")
    print(cos_w.shape)
    delta_w_angle = 1 - cos_w
    
    h_edges = train_mesh[:,0:grid_h,:,:] - train_mesh[:,1:grid_h+1,:,:]
    cos_h = tf.reduce_sum(h_edges[:,0:grid_h-1,:,:] * h_edges[:,1:grid_h,:,:],3) / (tf.sqrt(tf.reduce_sum(h_edges[:,0:grid_h-1,:,:]*h_edges[:,0:grid_h-1,:,:],3))*tf.sqrt(tf.reduce_sum(h_edges[:,1:grid_h,:,:]*h_edges[:,1:grid_h,:,:],3)))
    delta_h_angle = 1 - cos_h
    
    
    loss = tf.reduce_mean(delta_w_angle) + tf.reduce_mean(delta_h_angle)
    
    
    return loss  
   
