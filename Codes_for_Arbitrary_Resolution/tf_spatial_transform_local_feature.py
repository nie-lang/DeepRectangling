# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf
import numpy as np
import math
import tensorDLT_local
from keras.layers import UpSampling2D

import constant
grid_w = constant.GRID_W
grid_h = constant.GRID_H

def transformer(U, theta, name='SpatialTransformer', **kwargs):

    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(
                tf.expand_dims(tf.ones(shape=tf.stack([n_repeats, ])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        with tf.variable_scope('_interpolate'):
            # constants
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]

            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

            # scale indices from [-1, 1] to [0, width/height]
            #x = (x + 1.0)*(width_f) / 2.0
            #y = (y + 1.0)*(height_f) / 2.0

            # do sampling
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1

            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width*height
            base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
            base_y0 = base + y0*dim2
            base_y1 = base + y1*dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1

            # use indices to lookup pixels in the flat image and restore
            # channels dim
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)

            # and finally calculate interpolated values
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
            wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
            wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
            wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
            output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
            return output




    #input:  batch_size*(grid_h+1)*(grid_w+1)*2
    #output: batch_size*grid_h*grid_w*9
    def get_Hs(theta, width, height): 
        with tf.variable_scope('get_Hs'):
            num_batch = tf.shape(theta)[0]
            h = height / grid_h
            w = width / grid_w
            Hs = []
            for i in range(grid_h):
                for j in range(grid_w):
                    hh = i * h
                    ww = j * w
                    ori = tf.tile(tf.constant([ww, hh, ww + w, hh, ww, hh + h, ww + w, hh + h], shape=[1, 8], dtype=tf.float32), multiples=[num_batch, 1])
                    #id = i * (grid_w + 1) + grid_w
                    tar = tf.concat([tf.slice(theta, [0, i, j, 0], [-1, 1, 1, -1]), tf.slice(theta, [0, i, j + 1, 0], [-1, 1, 1, -1]), 
                    tf.slice(theta, [0, i + 1, j, 0], [-1, 1, 1, -1]), tf.slice(theta, [0, i + 1, j + 1, 0], [-1, 1, 1, -1])], axis=1)
                    tar = tf.reshape(tar, [num_batch, 8])
                    #tar = tf.Print(tar, [tf.slice(ori, [0, 0], [1, -1])],message="[ori--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                    #tar = tf.Print(tar, [tf.slice(tar, [0, 0], [1, -1])],message="[tar--i:"+str(i)+",j:"+str(j)+"]:", summarize=100,first_n=5)
                    Hs.append(tf.reshape(tensorDLT_local.solve_DLT(ori, tar), [num_batch, 1, 9]))   
            Hs = tf.reshape(tf.concat(Hs, axis=1), [num_batch, grid_h, grid_w, 9], name='Hs')
        return Hs 

    def _meshgrid2(height, width, sh, eh, sw, ew):
        hn = eh - sh + 1
        wn = ew - sw + 1
        
        
        x_t = tf.matmul(tf.ones(shape=tf.stack([hn, 1])),
                        tf.transpose(tf.expand_dims(tf.slice(tf.linspace(-1.0, 1.0, width), [sw], [wn]), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.slice(tf.linspace(-1.0, 1.0, height), [sh], [hn]), 1),
                        tf.ones(shape=tf.stack([1, wn])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
        return grid
        
    def _meshgrid(height, width):


        #x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
        #                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        #y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
        #                        tf.ones(shape=tf.stack([1, width])))
        x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])),
                                tf.transpose(tf.expand_dims(tf.linspace(0., tf.cast(width, 'float32')-1.001, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(0., tf.cast(height, 'float32')-1.001, height), 1),
                                tf.ones(shape=tf.stack([1, width])))                        

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat([x_t_flat, y_t_flat, ones], 0)
            
        return grid



    def _transform3(theta, input_dim):
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            
            # the width/height should be an an integral multiple of grid_w/grid_h
            width_float = 32.
            height_float = 24.
            #M = np.array([[width_float / 2.0, 0., width_float / 2.0],
            #      [0., height_float / 2.0, height_float / 2.0],
            #      [0., 0., 1.]]).astype(np.float32)
            #M_tensor = tf.constant(M, tf.float32)
            #M_tile = tf.tile(tf.expand_dims(M_tensor, [0]), [num_batch, 1, 1])
            #M_inv = np.linalg.inv(M)
            #M_tensor_inv = tf.constant(M_inv, tf.float32)
            #M_tile_inv = tf.tile(tf.expand_dims(M_tensor_inv, [0]), [num_batch, 1, 1])
    
            theta = tf.cast(theta, 'float32')
            Hs = get_Hs(theta, width_float, height_float)
            gh = tf.cast(height / grid_h, 'int32')
            gw =tf.cast(width / grid_w, 'int32')
            ##########################################
            print("Hs")
            print(Hs.shape)
            H_array = UpSampling2D(size=(24/grid_h, 32/grid_w))(Hs)
            H_array = tf.reshape(H_array, [-1, 3, 3])
            ##########################################
            
            
            out_height = height
            out_width = width
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))  # stack num_batch grids
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))
            print("grid")
            print(grid.shape)
            ### [bs, 3, N]
            
            
            grid = tf.expand_dims(tf.transpose(grid, [0, 2, 1]),3)
            ### [bs, 3, N] -> [bs, N, 3] -> [bs, N, 3, 1]
            grid = tf.reshape(grid, [-1, 3, 1])
            ### [bs*N, 3, 1]
            
            
            
            grid_row = tf.reshape(grid, [-1, 3])
            print("grid_row")
            print(grid_row.shape)
            x_s = tf.reduce_sum(tf.multiply(H_array[:,0,:], grid_row), 1)
            y_s = tf.reduce_sum(tf.multiply(H_array[:,1,:], grid_row), 1)
            t_s = tf.reduce_sum(tf.multiply(H_array[:,2,:], grid_row), 1)
            
            
            # The problem may be here as a general homo does not preserve the parallelism
            # while an affine transformation preserves it.
            t_s_flat = tf.reshape(t_s, [-1])
            t_1 = tf.ones(shape = tf.shape(t_s_flat))
            t_0 = tf.zeros(shape = tf.shape(t_s_flat))      
            sign_t = tf.where(t_s_flat >= 0, t_1, t_0) * 2 - 1
            t_s_flat = t_s_flat + sign_t*1e-8

            x_s_flat = tf.reshape(x_s, [-1]) / t_s_flat
            y_s_flat = tf.reshape(y_s, [-1]) / t_s_flat
            

            out_size = (height, width)
            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
            #mask_transformed = _interpolate(mask, x_s_flat, y_s_flat, out_size)

            warp_image = tf.reshape(input_transformed, tf.stack([num_batch, height, width, num_channels]), name='output_img')
           # warp_mask = tf.reshape(mask_transformed, tf.stack([num_batch, height, width, num_channels]), name='output_mask')

            

            return warp_image

    
    with tf.variable_scope(name):
        #output = _transform(theta, U, out_size)
        #U = U - 1.
        warp_image = _transform3(theta, U)
        #warp_image = warp_image + 1.
        #warp_image = tf.clip_by_value(warp_image, -1, 1)
        return warp_image


