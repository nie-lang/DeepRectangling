import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers import conv2d
import tf_spatial_transform_local_feature
import tf_mesh2flow
import constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H


# Warping layer ---------------------------------
def get_grid(x):
    batch_size, height, width, filters = tf.unstack(tf.shape(x))
    Bg, Yg, Xg = tf.meshgrid(tf.range(batch_size), tf.range(height), tf.range(width),
                             indexing = 'ij')
    # return indices volume indicate (batch, y, x)
    # return tf.stack([Bg, Yg, Xg], axis = 3)
    return Bg, Yg, Xg # return collectively for elementwise processing

def nearest_warp(x, flow):
    grid_b, grid_y, grid_x = get_grid(x)
    flow = tf.cast(flow, tf.int32)

    warped_gy = tf.add(grid_y, flow[:,:,:,1]) # flow_y
    warped_gx = tf.add(grid_x, flow[:,:,:,0]) # flow_x
    # clip value by height/width limitation
    _, h, w, _ = tf.unstack(tf.shape(x))
    warped_gy = tf.clip_by_value(warped_gy, 0, h-1)
    warped_gx = tf.clip_by_value(warped_gx, 0, w-1)
            
    warped_indices = tf.stack([grid_b, warped_gy, warped_gx], axis = 3)
            
    warped_x = tf.gather_nd(x, warped_indices)
    return warped_x

def bilinear_warp(x, flow):
    _, h, w, _ = tf.unstack(tf.shape(x))
    grid_b, grid_y, grid_x = get_grid(x)
    grid_b = tf.cast(grid_b, tf.float32)
    grid_y = tf.cast(grid_y, tf.float32)
    grid_x = tf.cast(grid_x, tf.float32)

    fx, fy = tf.unstack(flow, axis = -1)
    fx_0 = tf.floor(fx)
    fx_1 = fx_0+1
    fy_0 = tf.floor(fy)
    fy_1 = fy_0+1

    # warping indices
    h_lim = tf.cast(h-1, tf.float32)
    w_lim = tf.cast(w-1, tf.float32)
    gy_0 = tf.clip_by_value(grid_y + fy_0, 0., h_lim)
    gy_1 = tf.clip_by_value(grid_y + fy_1, 0., h_lim)
    gx_0 = tf.clip_by_value(grid_x + fx_0, 0., w_lim)
    gx_1 = tf.clip_by_value(grid_x + fx_1, 0., w_lim)
    
    g_00 = tf.cast(tf.stack([grid_b, gy_0, gx_0], axis = 3), tf.int32)
    g_01 = tf.cast(tf.stack([grid_b, gy_0, gx_1], axis = 3), tf.int32)
    g_10 = tf.cast(tf.stack([grid_b, gy_1, gx_0], axis = 3), tf.int32)
    g_11 = tf.cast(tf.stack([grid_b, gy_1, gx_1], axis = 3), tf.int32)

    # gather contents
    x_00 = tf.gather_nd(x, g_00)
    x_01 = tf.gather_nd(x, g_01)
    x_10 = tf.gather_nd(x, g_10)
    x_11 = tf.gather_nd(x, g_11)

    # coefficients
    c_00 = tf.expand_dims((fy_1 - fy)*(fx_1 - fx), axis = 3)
    c_01 = tf.expand_dims((fy_1 - fy)*(fx - fx_0), axis = 3)
    c_10 = tf.expand_dims((fy - fy_0)*(fx_1 - fx), axis = 3)
    c_11 = tf.expand_dims((fy - fy_0)*(fx - fx_0), axis = 3)

    return c_00*x_00 + c_01*x_01 + c_10*x_10 + c_11*x_11
    
def shift2mesh(mesh_shift, width, height):
    batch_size = tf.shape(mesh_shift)[0]
    h = height/grid_h
    w = width/grid_w
    ori_pt = []
    for i in range(grid_h + 1):
        for j in range(grid_w + 1):
            ww = j * w
            hh = i * h
            p = tf.constant([ww, hh], shape=[2], dtype=tf.float32)
            ori_pt.append(tf.expand_dims(p, 0))
    ori_pt = tf.concat(ori_pt, axis=0)
    ori_pt = tf.reshape(ori_pt, [grid_h+1, grid_w+1, 2])
    ori_pt = tf.tile(tf.expand_dims(ori_pt, 0),[batch_size, 1, 1, 1])

    tar_pt = ori_pt + mesh_shift
    #tar_pt = tf.reshape(tar_pt, [batch_size, grid_h+1, grid_w+1, 2])
    
    return tar_pt

def flow_resize_operation(flow_input, height, width):
    flow_tmp = tf.image.resize_images(flow_input, [height,width],method=1)
    flow_x = flow_tmp[:, :, :, 0] * tf.cast(width, tf.float32) /512.
    flow_y = flow_tmp[:, :, :, 1] * tf.cast(height, tf.float32) /384.
    flow_output = tf.stack([flow_x, flow_y], 3)
    return flow_output

def RectanglingNetwork(train_input, train_mask):
    
    resized_train_input = tf.image.resize_images(train_input, [384,512],method=0)
    resized_train_mask = tf.image.resize_images(train_mask, [384,512],method=0)
    
    batch_size = tf.shape(train_input)[0]
    height = tf.shape(train_input)[1]
    width = tf.shape(train_input)[2]
    
    resized_mesh_shift_primary, resized_mesh_shift_final = build_model(resized_train_input, resized_train_mask) 
    
    resized_mesh_primary = shift2mesh(resized_mesh_shift_primary, 512., 384.)
    resized_mesh_final = shift2mesh(resized_mesh_shift_final+resized_mesh_shift_primary, 512., 384.)
    resized_flow_final = tf_mesh2flow.mesh2flow(resized_mesh_final)
    flow_final = flow_resize_operation(resized_flow_final, height, width)
    warp_image_final = bilinear_warp(train_input, flow_final)
    
    return warp_image_final


# feature extraction module
def feature_extractor(image_tf):
    feature = []
    # 512*384
    with tf.variable_scope('conv_block1'): 
      conv1 = conv2d(inputs=image_tf, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
      conv1 = conv2d(inputs=conv1, num_outputs=64, kernel_size=3, rate=1, activation_fn=tf.nn.relu)
      maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')
    # 256*192
    with tf.variable_scope('conv_block2'):
      conv2 = conv2d(inputs=maxpool1, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
      conv2 = conv2d(inputs=conv2, num_outputs=64, kernel_size=3, activation_fn=tf.nn.relu)
      maxpool2 = slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')
    # 128*96
    with tf.variable_scope('conv_block3'):
      conv3 = conv2d(inputs=maxpool2, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      conv3 = conv2d(inputs=conv3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')
    # 64*48
    with tf.variable_scope('conv_block4'):
      conv4 = conv2d(inputs=maxpool3, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      conv4 = conv2d(inputs=conv4, num_outputs=128, kernel_size=3, activation_fn=tf.nn.relu)
      feature.append(conv4)
    
    return feature

    

# mesh motion regression module
def regression_Net(correlation):

    conv1 = conv2d(inputs=correlation, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv1 = conv2d(inputs=conv1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool1 = slim.max_pool2d(conv1, 2, stride=2, padding = 'SAME')     #16
    conv2 = conv2d(inputs=maxpool1, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    conv2 = conv2d(inputs=conv2, num_outputs=256, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool2 =slim.max_pool2d(conv2, 2, stride=2, padding = 'SAME')    #8
    conv3 = conv2d(inputs=maxpool2, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv3 = conv2d(inputs=conv3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    
    maxpool3 = slim.max_pool2d(conv3, 2, stride=2, padding = 'SAME')    #4
    conv4 = conv2d(inputs=maxpool3, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    conv4 = conv2d(inputs=conv4, num_outputs=512, kernel_size=3, activation_fn=tf.nn.relu)
    
    
    fc1 = conv2d(inputs=conv4, num_outputs=2048, kernel_size=[3,4], activation_fn=tf.nn.relu, padding="VALID")
    fc2 = conv2d(inputs=fc1, num_outputs=1024, kernel_size=1, activation_fn=tf.nn.relu)
    fc3 = conv2d(inputs=fc2, num_outputs=(grid_w+1)*(grid_h+1)*2, kernel_size=1, activation_fn=None)
    #net3_f = tf.expand_dims(tf.squeeze(tf.squeeze(fc3,1),1), [2])
    net3_f_local = tf.reshape(fc3, (-1, grid_h+1, grid_w+1, 2))
    
    return net3_f_local
    



def build_model(train_input, train_mask):
    with tf.variable_scope('model'):
      batch_size = tf.shape(train_input)[0]
    
      with tf.variable_scope('feature_extract', reuse = None): 
        features = feature_extractor(tf.concat([train_input, train_mask], axis=3))
    
      feature = tf.image.resize_images(features[-1], [24, 32], method=0)
      with tf.variable_scope('regression_coarse', reuse = None): 
        mesh_shift_primary = regression_Net(feature)
      
      with tf.variable_scope('regression_fine', reuse = None): 
        mesh_primary = shift2mesh(mesh_shift_primary/16, 32., 24.)
        feature_warp = tf_spatial_transform_local_feature.transformer(feature, mesh_primary)
        mesh_shift_final = regression_Net(feature_warp)
      
    
      return mesh_shift_primary, mesh_shift_final