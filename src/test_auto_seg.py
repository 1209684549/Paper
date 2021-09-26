# -*- coding: utf-8 -*-
"""
Created on Thu Sep  9 09:01:27 2021

@author: yangyue
"""

import sys
import glob
import gc
import apps as app
# third party
import tensorflow as tf
import numpy as np
from keras.backend.tensorflow_backend import set_session
#import dataprint
# project
sys.path.append('../ext/medipy-lib')

#import networks
from medipy.metrics import dice
import os
import data_gen
import nibabel as nib
import network_auto_seg


from write_excel import *

import networks
 
base_data_dir = '../data/oasis1/' 
postname='nii'

labels = np.array([1,2,3])

def test( gpu_id, vol_size=(64,64,64)):
    
    #GPU配置
    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    sess = tf.Session(config=config)
    set_session(sess)
    
    #测试图像列表
    test_names = glob.glob(base_data_dir + 'test/image1/*')
    test_names = sorted(test_names)
    
    test_seg_names = glob.glob(base_data_dir + 'test/label1/*')
    test_seg_names = sorted(test_seg_names)
    
    atlas_names = glob.glob(base_data_dir + 'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.nii.gz')
    atlasseg_data= glob.glob(base_data_dir + 'OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.nii.gz') 
    
    mode ='test'
    enc_nf = [16,32,64,128]
    dec_nf = [128,64,64,32,32,16,16]
    
    model_name = glob.glob('../model/*')
    patch_size = (64,64,64)
    model = network_auto_seg.auto_context_one_attunet_seg(vol_size,enc_nf,dec_nf,mode)

    stride = 32
    edgesize=3
    nponesmatminus_edge=np.ones(np.array([64,64,64]+[3])-np.array([2*edgesize,2*edgesize,2*edgesize,0]),dtype=np.float32)

    data_temp = nib.load(atlas_names[0])
    affine = data_temp.affine.copy()
    data_temp = data_temp.get_data()


    
    for a in range(len(model_name)):
        current_model =model_name[a]
        model.load_weights(current_model)
        
                    #图谱取patch
        atlas_vols, atlas_patch_loc = data_gen.single_vols_generator_patch(
                atlas_names[0],1,patch_size, stride_patch = stride,out =2)
      
        for j in range(len(test_names)):
            
            #测试数据集取patch
            test_vols, test_patch_loc = data_gen.single_vols_generator_patch(
                test_names[j],1,patch_size,stride_patch=stride,out=2)
            
            test_vols_seg, test_patch_loc = data_gen.single_vols_generator_patch(
                test_seg_names[j],1,patch_size,stride_patch=stride,out=2)
            

      
            
            mask1=np.empty(data_temp.shape+(3,)).astype('float32')
            mask2=np.empty(data_temp.shape+(3,)).astype('float32')
            
            patch_number = 80

        
            
            for i in range(patch_number):#the number of patches for any one image               
          
                _,_,_,_,_, pred_temp = model.predict([test_vols[i],atlas_vols[i],test_vols_seg[i]])    
                
                mask1[test_patch_loc[i][0].start+edgesize:test_patch_loc[i][0].stop-edgesize,
                test_patch_loc[i][1].start+edgesize:test_patch_loc[i][1].stop-edgesize,test_patch_loc[i][2].start+edgesize:test_patch_loc[i][2].stop-edgesize,:] += pred_temp[0,edgesize:-edgesize,edgesize:-edgesize,edgesize:-edgesize,:]
                mask2[test_patch_loc[i][0].start+edgesize:test_patch_loc[i][0].stop-edgesize,
                test_patch_loc[i][1].start+edgesize:test_patch_loc[i][1].stop-edgesize,test_patch_loc[i][2].start+edgesize:test_patch_loc[i][2].stop-edgesize,:] += nponesmatminus_edge
                
            
            total_ddf_j=mask1/(mask2 +np.finfo(float).eps)
            
            #保存形变场
            new_flow = nib.Nifti1Image(total_ddf_j, affine)
            nib.save(new_flow, '../print/'+str(a)+ str(j) + 'flow.nii')
           

            if postname=='npz':#取分割图像
               testlabel = np.load(test_seg_names[j])['vol_data']
               atlas_seg = np.load(atlasseg_data[0])['vol_data']
            else:
               testlabel = nib.load(test_seg_names[j]).get_fdata()
               atlas_seg = nib.load(atlasseg_data[0]).get_fdata()            
            
            #测试集分割图像处理
            testlabel = np.reshape(testlabel, testlabel.shape + (1,)).astype('float32')
           # testlabel = tf.convert_to_tensor(testlabel)
            
            #形变场处理
            flow = np.reshape(total_ddf_j, (1,)+ total_ddf_j.shape).astype('float32')
           
            
            warp_seg=app.warp_volumes_by_ddf_AD(testlabel, flow)   
            #保存形变后的分割图
            new_seg = nib.Nifti1Image(warp_seg[0,:,:,:,0], affine)
            nib.save(new_seg, '../print/'+str(a) + str(j) + 'seg.nii')
           
            
    
                
            vals, _ = dice(warp_seg[0,:,:,:,0], atlas_seg, labels, nargout=2)
            a0 = vals[0]
            a1 = vals[1]
            a2 = vals[2]
            dice_mean = np.mean(vals)
            dice_std = np.std(vals)
        #    print('vals:',vals)
            print('Dice mean over structures: {:.2f} ({:.2f})'.format(dice_mean, dice_std))
         
            print(test_seg_names[j])
            b = [[a0,a1,a2,np.mean(vals),np.std(vals)]]
           # print(b)
            write_excel_xls_append('../xlsx/1.xlsx',b)
            del total_ddf_j,flow,testlabel,atlas_seg,warp_seg
            gc.collect()
     


    
    
if __name__ == "__main__":        
    test(0)


