# python imports
import os
import glob
import sys
from argparse import ArgumentParser
import data_gen
# third-party imports
import tensorflow as tf
import numpy as np

import os
import matplotlib.pyplot as plt

from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

# project imports
sys.path.append('../ext/medipy-lib')

import network_auto_seg
import losses
from tensorboardX import SummaryWriter

writer = SummaryWriter('../logs1/')
vol_size = (64,64,64) # (160, 192, 160)   patch大小

base_data_dir = '../data/oasis1/'

def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter):
    model_dir = '../model1/'    #模型保存目录
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)

    gpu = '/gpu:' + str(gpu_id)     #GPU配置
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto() 
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))
    
    nf_enc = [16, 32, 64, 128]
    nf_dec = [128, 64, 64, 32, 32, 16, 16]
    
    data_loss  = losses.NCC_SSIM3D().loss        
    data_loss5 = losses.NCC_SSIM3D(win = [5,5,5]).loss 
    dice_local  = losses.dice_loss_local().loss
    dice_loss= losses.dice_coe()



    with tf.device(gpu):   #GPU

        model = network_auto_seg.auto_context_one_attunet_seg(vol_size,nf_enc,nf_dec)

        model.compile(optimizer=Adam(lr=lr),                          
                      loss=[data_loss,dice_local,losses.Grad('l2').loss,data_loss5,losses.Grad('l2').loss,data_loss5,dice_loss,losses.Grad('l2').loss],
                      loss_weights=[0.1,0.15,0.05,0.1,0.05,0.3,0.15,0.1])
        model.summary()
        
        

    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))
    ####new code to use patch to train
    
    stride = 32
    patch_size = [64,64,64]   
    
    '''  图谱  '''
    atlas = base_data_dir+'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii.gz'
    atlas_seg = base_data_dir + 'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.nii.gz'

    '''数据集划分'''
    train_vol_names = glob.glob(base_data_dir + 'train/*')
    train_vol_names_seg = glob.glob(base_data_dir + 'train_seg/*')


    
    num_training =len(train_vol_names)
    number  = 1

    
    for step in range(0, n_iterations):
        atlas_patch = data_gen.single_vols_generator_patch(atlas,
                                    num_data = number ,patch_size = patch_size,stride_patch = stride)
        atlas_patch_seg = data_gen.single_vols_generator_patch(atlas_seg,
                                    num_data = number ,patch_size = patch_size,stride_patch = stride)
        for k in range(num_training):
            train_patch =  data_gen.single_vols_generator_patch(vol_name = train_vol_names[k], 
                                    num_data = number ,patch_size = patch_size,stride_patch = stride)
            train_patch_seg =  data_gen.single_vols_generator_patch(vol_name = train_vol_names_seg[k], 
                                    num_data = number ,patch_size = patch_size,stride_patch = stride)
            patch_len = len(train_patch)
            for counter in range(patch_len):
                X = train_patch[counter]
                X = np.reshape(X, (1,) + X.shape)

                X_seg = train_patch_seg[counter]
                X_seg = np.reshape(X_seg, (1,) + X_seg.shape)
                
                atlas_vol = atlas_patch[counter]
                atlas_vol = np.reshape(atlas_vol, (1,) + atlas_vol.shape)
                
                atlas_vol_seg = atlas_patch_seg[counter]
                atlas_vol_seg = np.reshape(atlas_vol_seg, (1,) + atlas_vol_seg.shape)
    

            
                train_loss = model.train_on_batch([X, atlas_vol,X_seg], 
                                [atlas_vol,atlas_vol_seg,zero_flow,
                                 atlas_vol,zero_flow,
                                 atlas_vol,atlas_vol_seg,zero_flow])
                 
         
                if not isinstance(train_loss, list):
                   train_loss = [train_loss]
                    

                    
                printLoss(step, k, train_loss)
                model.save(model_dir + '/' + str(step) + '.h5')
        
    
        if(step % model_save_iter == 0):
            
            writer.add_scalar('0', train_loss[0],step)
            writer.add_scalar('1', train_loss[1],step)
            writer.add_scalar('2', train_loss[2],step)
            writer.add_scalar('3', train_loss[3],step)
            writer.add_scalar('4', train_loss[3],step)
            writer.add_scalar('5', train_loss[3],step)            
            writer.add_scalar('6', train_loss[3],step)
            writer.add_scalar('7', train_loss[3],step)
                


def printLoss(step, training, train_loss):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
        
    else:
        s += "," + str(train_loss)

    print(s)
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm1',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--save_name", type=str,# required=True,
                        dest="save_name", default='savename',
                        help="Name of model when saving")
    parser.add_argument("--gpu", type=int,default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,help="learning rate") 
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=60000,#150000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=1,#5000,
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args)) 
    
