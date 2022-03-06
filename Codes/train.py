import tensorflow as tf
import os

from model import RectanglingNetwork
from loss_functions import intensity_loss, intra_grid_loss, inter_grid_loss
from utils import load, save, DataLoader
from PIL import Image
import numpy as np
import scipy.io


import constant
os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = constant.GPU

train_folder = constant.TRAIN_FOLDER
test_folder = constant.TEST_FOLDER

batch_size = constant.TRAIN_BATCH_SIZE
iterations = constant.ITERATIONS

height, width = 384, 512

summary_dir = constant.SUMMARY_DIR
snapshot_dir = constant.SNAPSHOT_DIR


def build_net(ntype,nin,nwb=None,name=None):
    if ntype=='conv':
        return tf.nn.relu(tf.nn.conv2d(nin,nwb[0],strides=[1,1,1,1],padding='SAME',name=name)+nwb[1])
    elif ntype=='pool':
        return tf.nn.avg_pool(nin,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

def get_weight_bias(vgg_layers,i):
    weights=vgg_layers[i][0][0][2][0][0]
    weights=tf.constant(weights)
    bias=vgg_layers[i][0][0][2][0][1]
    bias=tf.constant(np.reshape(bias,(bias.size)))
    return weights,bias

vgg_path=scipy.io.loadmat('./vgg19/imagenet-vgg-verydeep-19.mat')
print("[i] Loaded pre-trained vgg19 parameters")

# build VGG19 to load pre-trained parameters
def build_vgg19(input,reuse=False):
    with tf.variable_scope("vgg19"):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        net={}
        vgg_layers=vgg_path['layers'][0]
        net['input']=input-np.array([123.6800, 116.7790, 103.9390]).reshape((1,1,1,3))
        net['conv1_1']=build_net('conv',net['input'],get_weight_bias(vgg_layers,0),name='vgg_conv1_1')
        net['conv1_2']=build_net('conv',net['conv1_1'],get_weight_bias(vgg_layers,2),name='vgg_conv1_2')
        net['pool1']=build_net('pool',net['conv1_2'])
        net['conv2_1']=build_net('conv',net['pool1'],get_weight_bias(vgg_layers,5),name='vgg_conv2_1')
        net['conv2_2']=build_net('conv',net['conv2_1'],get_weight_bias(vgg_layers,7),name='vgg_conv2_2')
        net['pool2']=build_net('pool',net['conv2_2'])
        net['conv3_1']=build_net('conv',net['pool2'],get_weight_bias(vgg_layers,10),name='vgg_conv3_1')
        net['conv3_2']=build_net('conv',net['conv3_1'],get_weight_bias(vgg_layers,12),name='vgg_conv3_2')
        net['conv3_3']=build_net('conv',net['conv3_2'],get_weight_bias(vgg_layers,14),name='vgg_conv3_3')
        net['conv3_4']=build_net('conv',net['conv3_3'],get_weight_bias(vgg_layers,16),name='vgg_conv3_4')
        net['pool3']=build_net('pool',net['conv3_4'])
        net['conv4_1']=build_net('conv',net['pool3'],get_weight_bias(vgg_layers,19),name='vgg_conv4_1')
        net['conv4_2']=build_net('conv',net['conv4_1'],get_weight_bias(vgg_layers,21),name='vgg_conv4_2')
        net['conv4_3']=build_net('conv',net['conv4_2'],get_weight_bias(vgg_layers,23),name='vgg_conv4_3')
        net['conv4_4']=build_net('conv',net['conv4_3'],get_weight_bias(vgg_layers,25),name='vgg_conv4_4')
        net['pool4']=build_net('pool',net['conv4_4'])
        net['conv5_1']=build_net('conv',net['pool4'],get_weight_bias(vgg_layers,28),name='vgg_conv5_1')
        net['conv5_2']=build_net('conv',net['conv5_1'],get_weight_bias(vgg_layers,30),name='vgg_conv5_2')
        print(type(net))
        return net




# define dataset
with tf.name_scope('dataset'):
    ##########training###############
    ###input###
    train_data_loader = DataLoader(train_folder)
    train_data_dataset = train_data_loader(batch_size=batch_size)
    train_data_it = train_data_dataset.make_one_shot_iterator()
    train_input_tensor = train_data_it.get_next()
    train_input_tensor.set_shape([batch_size, height, width, 9])
    
    train_input = train_input_tensor[:,:,:,0:3]
    train_mask = train_input_tensor[:,:,:,3:6]
    train_gt = train_input_tensor[:,:,:,6:9]
    
    print('train input = {}'.format(train_input))
    print('train mask = {}'.format(train_mask))
    print('train gt = {}'.format(train_gt))



# define training rectangling function
with tf.variable_scope('generator', reuse=None):
    print('training = {}'.format(tf.get_variable_scope().name))
    train_mesh_primary, train_warp_image_primary, train_warp_mask_primary, train_mesh_final, train_warp_image_final, train_warp_mask_final = RectanglingNetwork(train_input, train_mask)
   

# define appearance loss (loss 1 of of the content term)
lam_appearance = 1
if lam_appearance != 0:
    primary_appearance_loss = intensity_loss(gen_frames=train_warp_image_primary, gt_frames=train_gt, l_num=1)
    final_appearance_loss = intensity_loss(gen_frames=train_warp_image_final, gt_frames=train_gt, l_num=1)
    appearance_loss = primary_appearance_loss + final_appearance_loss
else:
    primary_appearance_loss = tf.constant(0.0, dtype=tf.float32)
    final_appearance_loss = tf.constant(0.0, dtype=tf.float32)
    appearance_loss = primary_appearance_loss + final_appearance_loss

# define perception loss (loss 2 of of the content term)
primary_features_warp = build_vgg19((train_warp_image_primary+1)*127.5, reuse=False)
final_features_warp = build_vgg19((train_warp_image_final+1)*127.5, reuse=True)
features_gt = build_vgg19((train_gt+1)*127.5, reuse=True)
lam_perception = 5e-6
if lam_perception != 0:  
    primary_perception_loss = intensity_loss(gen_frames=primary_features_warp['conv4_2'], gt_frames=features_gt['conv4_2'], l_num=2)
    final_perception_loss = intensity_loss(gen_frames=final_features_warp['conv4_2'], gt_frames=features_gt['conv4_2'], l_num=2)
    perception_loss = primary_perception_loss + final_perception_loss
else:
    primary_perception_loss = tf.constant(0.0, dtype=tf.float32)
    final_perception_loss = tf.constant(0.0, dtype=tf.float32)
    perception_loss = primary_perception_loss + final_perception_loss

# define boundary term
lam_mask = 1
if lam_mask != 0:
    primary_mask_loss = intensity_loss(gen_frames=train_warp_mask_primary, gt_frames=tf.ones_like(train_warp_mask_primary), l_num=1)
    final_mask_loss = intensity_loss(gen_frames=train_warp_mask_final, gt_frames=tf.ones_like(train_warp_mask_final), l_num=1)
    mask_loss = primary_mask_loss + final_mask_loss
else:
    primary_mask_loss = tf.constant(0.0, dtype=tf.float32)
    final_mask_loss = tf.constant(0.0, dtype=tf.float32)
    mask_loss = primary_mask_loss + final_mask_loss

# define mesh term
lam_mesh = 1
if lam_mesh != 0:
    primary_mesh_loss = intra_grid_loss(train_mesh_primary) + inter_grid_loss(train_mesh_primary)
    final_mesh_loss = intra_grid_loss(train_mesh_final) + inter_grid_loss(train_mesh_final)
    mesh_loss = primary_mesh_loss + final_mesh_loss
else:
    primary_mesh_loss = tf.constant(0.0, dtype=tf.float32)
    final_mesh_loss = tf.constant(0.0, dtype=tf.float32)
    mesh_loss = primary_mesh_loss + final_mesh_loss


with tf.name_scope('training'):
    g_loss = tf.add_n([appearance_loss * lam_appearance, perception_loss * lam_perception, lam_mask * mask_loss, mesh_loss * lam_mesh], name='g_loss')

    g_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='g_step')
    g_lrate = tf.train.exponential_decay(0.0001, g_step, decay_steps=50000/4, decay_rate=0.96)
    g_optimizer = tf.train.AdamOptimizer(learning_rate=g_lrate, name='g_optimizer')
    g_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')
    
    grads = g_optimizer.compute_gradients(g_loss, var_list=g_vars)
    for i, (g, v) in enumerate(grads):
      if g is not None:
        grads[i] = (tf.clip_by_norm(g, 3), v)  # clip gradients
    g_train_op = g_optimizer.apply_gradients(grads, global_step=g_step, name='g_train_op')
    

# add all to summaries'
#loss
tf.summary.scalar(tensor=g_loss, name='g_loss')
tf.summary.scalar(tensor=appearance_loss, name='appearance_loss')
tf.summary.scalar(tensor=perception_loss, name='perception_loss')
tf.summary.scalar(tensor=mask_loss, name='mask_loss')
tf.summary.scalar(tensor=mesh_loss, name='mesh_loss')
#images
tf.summary.image(tensor=train_input, name='train_input')
tf.summary.image(tensor=train_mask, name='train_mask')
tf.summary.image(tensor=train_gt, name='train_gt')
tf.summary.image(tensor=train_warp_image_primary, name='train_warp_image_primary')
tf.summary.image(tensor=train_warp_image_final, name='train_warp_image_final')



summary_op = tf.summary.merge_all()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    # summaries
    summary_writer = tf.summary.FileWriter(summary_dir, graph=sess.graph)

    # initialize weights
    sess.run(tf.global_variables_initializer())
    print('Init successfully!')

    # tf saver
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=None)
    restore_var = [v for v in tf.global_variables()]
    loader = tf.train.Saver(var_list=restore_var)
    print("snapshot_dir")
    print(snapshot_dir)
    if os.path.isdir(snapshot_dir):
        ckpt = tf.train.get_checkpoint_state(snapshot_dir)
        if ckpt and ckpt.model_checkpoint_path:
            load(loader, sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found.')
    else:
        load(loader, sess, snapshot_dir)

    _step, _loss, _summaries = 0, None, None

    print("============starting training===========")
    while _step < iterations:
        try:
            print('Training generator...')
            _, _g_lr, _step, _appearance_loss, _perception_loss, _mask_loss, _mesh_loss, _g_loss, _summaries = sess.run([g_train_op, g_lrate, g_step, appearance_loss, perception_loss, mask_loss, mesh_loss, g_loss, summary_op])

            if _step % 100 == 0:
                print('GeneratorModel : Step {}, lr = {:.8f}'.format(_step, _g_lr))
                print('                 Global      Loss : ', _g_loss)
                print('                 appearance   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_appearance_loss, lam_appearance, _appearance_loss * lam_appearance))
                print('                 vgg   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_perception_loss, lam_perception, _perception_loss * lam_perception))
                print('                 mask   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_mask_loss, lam_mask, _mask_loss * lam_mask))
                print('                 mesh   Loss : ({:.4f} * {:.4f} = {:.4f})'.format(_mesh_loss, lam_mesh, _mesh_loss * lam_mesh))
            if _step % 2000 == 0:
                summary_writer.add_summary(_summaries, global_step=_step)
                print('Save summaries...')

            if _step % 100000 == 0:
                save(saver, sess, snapshot_dir, _step)

        except tf.errors.OutOfRangeError:
            print('Finish successfully!')
            save(saver, sess, snapshot_dir, _step)
            break
