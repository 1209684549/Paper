"""
data generators for VoxelMorph

for the CVPR and MICCAI papers, we have data arranged in train/validate/test folders
inside each folder is a /vols/ and a /asegs/ folder with the volumes
and segmentations. All of our papers use npz formated data.
"""

import os, sys
import numpy as np
import nibabel as nib
sys.path.append('../ext')
import labelreg.helpers as helper
import labelreg.utils1 as util1
#import labelreg.utils as util1

def cvpr2018_gen(gen, atlas_vol_bs, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs0 = (util1.warp_image_affine(atlas_vol_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            yield ([X, atlas_vol_bs0], [atlas_vol_bs0, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])
def cvpr2018_gen_seg(gen, atlas_vol1_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,seg] = next(gen)
#        seg3 = np.concatenate((atlas_seg,seg),axis=-1)
        idx = np.random.randint(3)
        seg = seg[...,idx]
        seg = seg[..., np.newaxis]
        X = atlas_seg[...,idx]
        X = X[..., np.newaxis]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, seg, X], [zeros,atlas_vol1_bs0,zeros])
        else:
            yield ([X1, atlas_vol1_bs, seg, X], [zeros,atlas_vol1_bs,zeros])
def cvpr2018_gen_newseg(gen, atlas_vol1_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,seg] = next(gen)
        idx = np.random.randint(3)
        seg = seg[...,idx]
        seg = seg[..., np.newaxis]
        X = atlas_seg[...,idx]
        X = X[..., np.newaxis]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, seg, X], [zeros,zeros])
        else:
            yield ([X1, atlas_vol1_bs, seg, X], [zeros,zeros])
#        yield ([X1, atlas_vol1_bs, seg, X], [])
def cvpr2018_gen_seg3(gen, atlas_vol1_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,seg] = next(gen)
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            atlas_seg0,atlas_seg1,atlas_seg2= np.split(atlas_seg,3,axis=4)
            atlas_seg0 = util1.warp_image_affine(atlas_seg0, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg1 = util1.warp_image_affine(atlas_seg1, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg2 = util1.warp_image_affine(atlas_seg2, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg0 = np.concatenate((atlas_seg0,atlas_seg1,atlas_seg2),axis=4)
            seg0,seg1,seg2= np.split(seg,3,axis=4)
            seg0 = util1.warp_image_affine(seg0, ph_moving_affine).astype('float32')  # data augmentation
            seg1 = util1.warp_image_affine(seg1, ph_moving_affine).astype('float32')  # data augmentation
            seg2 = util1.warp_image_affine(seg2, ph_moving_affine).astype('float32')  # data augmentation
            seg = np.concatenate((seg0,seg1,seg2),axis=4)
            yield ([X1, atlas_vol1_bs0, atlas_seg0], [seg, zeros])
        else:
            yield ([X1, atlas_vol1_bs, atlas_seg], [seg, zeros])

def regi_net_segnew3_gen(gen, atlas_vol1_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
#    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    zeros = np.zeros((batch_size, 1, 1))
    while True:
        [X1,seg] = next(gen)
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            seg0,seg1,seg2= np.split(seg,3,axis=4)
            seg0 = util1.warp_image_affine(seg0, ph_moving_affine).astype('float32')  # data augmentation
            seg1 = util1.warp_image_affine(seg1, ph_moving_affine).astype('float32')  # data augmentation
            seg2 = util1.warp_image_affine(seg2, ph_moving_affine).astype('float32')  # data augmentation
            seg = np.concatenate((seg0,seg1,seg2),axis=4)
            atlas_seg0,atlas_seg1,atlas_seg2= np.split(atlas_seg,3,axis=4)
            atlas_seg0 = util1.warp_image_affine(atlas_seg0, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg1 = util1.warp_image_affine(atlas_seg1, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg2 = util1.warp_image_affine(atlas_seg2, ph_moving_affine).astype('float32')  # data augmentation
            atlas_seg_aug = np.concatenate((atlas_seg0,atlas_seg1,atlas_seg2),axis=4)
            yield ([X1, atlas_vol1_bs0, seg], [atlas_vol1_bs0,atlas_vol1_bs0,atlas_vol1_bs0,atlas_vol1_bs0,atlas_seg_aug,atlas_seg_aug,atlas_seg_aug,atlas_seg_aug,zeros,zeros,zeros,zeros])
        else:
            yield ([X1, atlas_vol1_bs, seg], [atlas_vol1_bs,atlas_seg,zeros,zeros,zeros])

def cvpr2018_gen_2channel_seg3(gen, atlas_vol1_bs, atlas_vol2_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,X2,seg] = next(gen)
        idx = np.random.randint(3)
        seg = seg[...,idx]
        seg = seg[..., np.newaxis]
        X = atlas_seg[...,idx]
        X = X[..., np.newaxis]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            atlas_vol2_bs0 = (util1.warp_image_affine(atlas_vol2_bs, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, X2, atlas_vol2_bs0, seg, X], [zeros,atlas_vol1_bs0, atlas_vol2_bs0, zeros])
        else:
            yield ([X1, atlas_vol1_bs, X2, atlas_vol2_bs, seg, X], [zeros,atlas_vol1_bs, atlas_vol2_bs, zeros])
def cvpr2018_gen_2channel(gen, atlas_vol1_bs, atlas_vol2_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,X2,seg] = next(gen)
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            atlas_vol2_bs0 = (util1.warp_image_affine(atlas_vol2_bs, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            atlas_seg0 = (util1.warp_image_affine(atlas_seg, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, X2, atlas_vol2_bs0, seg], [atlas_vol1_bs0, atlas_vol2_bs0, atlas_seg0, zeros, atlas_vol1_bs0, atlas_vol2_bs0])
        else:
            yield ([X1, atlas_vol1_bs, X2, atlas_vol2_bs, seg], [atlas_vol1_bs, atlas_vol2_bs, atlas_seg, zeros, atlas_vol1_bs, atlas_vol2_bs])
#model = Model(inputs=[src1, tgt1, src2, tgt2, seg], outputs=[warped_src1, warped_src2, w_seg, flow, warped_src1_byflow1, warped_src2_byflow2])
def cvpr2018_gen_2channel1(gen, atlas_vol1_bs, atlas_vol2_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,X2,seg] = next(gen)
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            atlas_vol2_bs0 = (util1.warp_image_affine(atlas_vol2_bs, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            atlas_seg0 = (util1.warp_image_affine(atlas_seg, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, X2, atlas_vol2_bs0, seg], [atlas_seg0, zeros])
        else:
            yield ([X1, atlas_vol1_bs, X2, atlas_vol2_bs, seg], [atlas_seg, zeros])
def cvpr2018_gen_2channel2(gen, atlas_vol1_bs, atlas_vol2_bs, atlas_seg, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """
    volshape = atlas_vol1_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1,X2,seg] = next(gen)
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol1_bs0 = (util1.warp_image_affine(atlas_vol1_bs, ph_moving_affine)).astype('float32')
            atlas_vol2_bs0 = (util1.warp_image_affine(atlas_vol2_bs, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            atlas_seg0 = (util1.warp_image_affine(atlas_seg, ph_moving_affine)).astype('float32')
            seg = (util1.warp_image_affine(seg, ph_moving_affine)).astype('float32')
            yield ([X1, atlas_vol1_bs0, X2, atlas_vol2_bs0, seg], [atlas_vol1_bs0, atlas_vol2_bs0, atlas_seg0, zeros])
        else:
            yield ([X1, atlas_vol1_bs, X2, atlas_vol2_bs, seg], [atlas_vol1_bs, atlas_vol2_bs, atlas_seg, zeros])
		
def autocontext_one_gen(gen, atlas_vol_bs, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs0 = (util1.warp_image_affine(atlas_vol_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            yield ([X, atlas_vol_bs0], [atlas_vol_bs0,atlas_vol_bs0,atlas_vol_bs0, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs,atlas_vol_bs,atlas_vol_bs, zeros])
def autocontext_one_3gen(gen, atlas_vol_bs, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model """

    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs0 = (util1.warp_image_affine(atlas_vol_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            yield ([X, atlas_vol_bs0], [atlas_vol_bs0,atlas_vol_bs0,atlas_vol_bs0,atlas_vol_bs0, zeros])
        else:
            yield ([X, atlas_vol_bs], [atlas_vol_bs,atlas_vol_bs,atlas_vol_bs,atlas_vol_bs, zeros])


def cvpr2018_gen_s2s(gen, batch_size=1, data_augment=True):
    """ generator used for cvpr 2018 model for subject 2 subject registration """
    zeros = None
    while True:
        X1 = next(gen)[0]
        X2 = next(gen)[0]

        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if data_augment: 
                ph_moving_affine=helper.random_transform_generator(batch_size)
                X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
                X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
        yield ([X1, X2], [X2, zeros])


def miccai2018_gen(gen, atlas_vol_bs, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs0 = (util1.warp_image_affine(atlas_vol_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            if bidir:
                yield ([X, atlas_vol_bs0], [atlas_vol_bs0, X, zeros])
            else:
                yield ([X, atlas_vol_bs0], [atlas_vol_bs0, zeros])
        else:
            if bidir:
                yield ([X, atlas_vol_bs], [atlas_vol_bs, X, zeros])
            else:
                yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros])
def autocontextdiff_one_gen(gen, atlas_vol_bs, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        X = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs0 = (util1.warp_image_affine(atlas_vol_bs, ph_moving_affine)).astype('float32')
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
            if bidir:
                yield ([X, atlas_vol_bs0], [atlas_vol_bs0, zeros, atlas_vol_bs0, X, zeros, atlas_vol_bs0])
            else:
                yield ([X, atlas_vol_bs0], [atlas_vol_bs0, zeros, atlas_vol_bs0, zeros, atlas_vol_bs0])
        else:
            if bidir:
                yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros, atlas_vol_bs, X, zeros, atlas_vol_bs])
            else:
                yield ([X, atlas_vol_bs], [atlas_vol_bs, zeros, atlas_vol_bs, zeros, atlas_vol_bs])

def miccai2018_t1t2_gen(gen, atlas_vol_bs1, atlas_vol_bs2, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs1.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1, X2] = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs10 = (util1.warp_image_affine(atlas_vol_bs1, ph_moving_affine)).astype('float32')
            atlas_vol_bs20 = (util1.warp_image_affine(atlas_vol_bs2, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            if bidir:
                yield ([X1, X2, atlas_vol_bs10, atlas_vol_bs20], [atlas_vol_bs10, atlas_vol_bs20, X1, X2, zeros])
            else:
                yield ([X1, X2, atlas_vol_bs10, atlas_vol_bs20], [atlas_vol_bs10, atlas_vol_bs20, zeros])
        else:
            if bidir:
                yield ([X1, X2, atlas_vol_bs1, atlas_vol_bs2], [atlas_vol_bs1, atlas_vol_bs2, X1, X2, zeros])
            else:
                yield ([X1, X2, atlas_vol_bs1, atlas_vol_bs2], [atlas_vol_bs1, atlas_vol_bs2, zeros])
def miccai2018_gen_2channel(gen, atlas_vol_bs1, atlas_vol_bs2, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    volshape = atlas_vol_bs1.shape[1:-1]
    zeros = np.zeros((batch_size, *volshape, len(volshape)))
    while True:
        [X1, X2] = next(gen)[0]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size)
            atlas_vol_bs10 = (util1.warp_image_affine(atlas_vol_bs1, ph_moving_affine)).astype('float32')
            atlas_vol_bs20 = (util1.warp_image_affine(atlas_vol_bs2, ph_moving_affine)).astype('float32')
            X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
            X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
            if bidir:
                yield ([X1, atlas_vol_bs10, X2, atlas_vol_bs20], [atlas_vol_bs10, atlas_vol_bs20, X1, X2, zeros])
            else:
                yield ([X1, atlas_vol_bs10, X2, atlas_vol_bs20], [atlas_vol_bs10, atlas_vol_bs20, zeros])
        else:
            if bidir:
                yield ([X1, atlas_vol_bs1, X2, atlas_vol_bs2], [atlas_vol_bs1, atlas_vol_bs2, X1, X2, zeros])
            else:
                yield ([X1, atlas_vol_bs1, X2, atlas_vol_bs2], [atlas_vol_bs1, atlas_vol_bs2, zeros])

def miccai2018_gen_s2s(gen, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        X = next(gen)[0]
        Y = next(gen)[0]
        if zeros is None:
            volshape = X.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if data_augment: 
                ph_moving_affine=helper.random_transform_generator(batch_size)
                X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')
                Y = (util1.warp_image_affine(Y, ph_moving_affine)).astype('float32')
        if bidir:
            yield ([X, Y], [Y, X, zeros])
        else:
            yield ([X, Y], [Y, zeros])

def miccai2018_gen_t1t2_s2s(gen, batch_size=1, bidir=False, data_augment=True):
    """ generator used for miccai 2018 model """
    zeros = None
    while True:
        [X1, X2] = next(gen)[0]
        [Y1, Y2] = next(gen)[0]
        if zeros is None:
            volshape = X1.shape[1:-1]
            zeros = np.zeros((batch_size, *volshape, len(volshape)))
        if data_augment: 
                ph_moving_affine=helper.random_transform_generator(batch_size)
                Y1 = (util1.warp_image_affine(Y1, ph_moving_affine)).astype('float32')
                X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')
                X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')
                Y2 = (util1.warp_image_affine(Y2, ph_moving_affine)).astype('float32')
        if bidir:
            yield ([X1,X2,Y1,Y2], [Y1,Y2,X1,X2,zeros])
        else:
            yield ([X1,X2,Y1,Y2], [Y1,Y2,zeros])


def load_testexample_by_name_nii(vol_names, return_T2=False, return_segs=False,return_seg3=False, return_hippo = False, np_var='vol_data'):
    idx = -1
    while True:
        idx = (idx+1)%len(vol_names)
        return_vals = []
        X = load_volfile(vol_names[idx], np_var=np_var)
        X = X[np.newaxis, ..., np.newaxis]
#        if data_augment: 
#            ph_moving_affine=helper.random_transform_generator(1,0.00001)
##           ph_moving_affine=helper.random_transform_generator(1)
#            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals.append(X)
        
        if return_T2:
            X = load_volfile(vol_names[idx].replace('T1_', 'T2_'), np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
#            if data_augment: 
#                X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
            return_vals.append(X)
        # also return segmentations
        volname = vol_names[idx].replace('-denoise-r-h', '')
        if return_segs:
            segname = volname.replace('T1_', 'matter_')
            X = load_volfile(segname, np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
#            if data_augment: 
#                X = util1.warp_image_affine(X, ph_moving_affine).astype('int16')  # data augmentation
            return_vals.append(X)
        if return_seg3:
            segname = volname.replace('T1_', 'seg3_')
            X = load_volfile(segname, np_var=np_var)
            #X = X[np.newaxis, ..., np.newaxis]
            X = X[np.newaxis, ...]
#            X0,X1,X2= np.split(X,3,axis=4)
#            if data_augment: 
#                X0 = util1.warp_image_affine(X0, ph_moving_affine).astype('int16')  # data augmentation
#                X1 = util1.warp_image_affine(X1, ph_moving_affine).astype('int16')  # data augmentation
#                X2 = util1.warp_image_affine(X2, ph_moving_affine).astype('int16')  # data augmentation
#            X = np.concatenate((X0,X1,X2),axis=4)
            return_vals.append(X)
        if return_hippo:
            segname = volname.replace('T1_', 'hippo_')
            X = load_volfile(segname, np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
#            if data_augment: 
#                X = util1.warp_image_affine(X, ph_moving_affine).astype('int16')  # data augmentation
            return_vals.append(X)
        yield tuple(return_vals)

def example_OASIS_gen(vol_names, batch_size=1, data_augment=False, return_T2=False, return_segs=False,return_seg3=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)
    
        X_data = []
        X_data2 = []
        X_seg = []
        X_seg3 = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            if data_augment: 
                ph_moving_affine=helper.random_transform_generator(batch_size,0.1)
#                ph_moving_affine=helper.random_transform_generator(1)
                X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
            X_data.append(X)
            
            if return_T2:
                X = load_volfile(vol_names[idx].replace('T1_', 'T2_'), np_var=np_var)
                X = X[np.newaxis, ..., np.newaxis]
                if data_augment: 
                    X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
                X_data2.append(X)
            # also return segmentations
            if return_segs:
                segname = vol_names[idx].replace('gfc', 'gfc_fseg')
                X = load_volfile(segname, np_var=np_var)
                X = X[np.newaxis, ..., np.newaxis]
                if data_augment: 
                    X = util1.warp_image_affine(X, ph_moving_affine).astype('float32')  # data augmentation
                X_seg.append(X)
            if return_seg3:
                segname = vol_names[idx].replace('gfc', 'gfc_segnew3_')
                X = load_volfile(segname, np_var=np_var)
                #X = X[np.newaxis, ..., np.newaxis]
                X = X[np.newaxis, ...]
                if data_augment: 
                    X0,X1,X2= np.split(X,3,axis=4)
                    X0 = util1.warp_image_affine(X0, ph_moving_affine).astype('float32')  # data augmentation
                    X1 = util1.warp_image_affine(X1, ph_moving_affine).astype('float32')  # data augmentation
                    X2 = util1.warp_image_affine(X2, ph_moving_affine).astype('float32')  # data augmentation
                    X = np.concatenate((X0,X1,X2),axis=4)
                X_seg3.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]
        if return_T2:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data2, 0))
            else:
                return_vals.append(X_data2[0])
        if return_segs:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_seg, 0))
            else:
                return_vals.append(X_seg[0])
        if return_seg3:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_seg3, 0))
            else:
                return_vals.append(X_seg3[0])
    
        yield tuple(return_vals)
def example_gen(vol_names, batch_size=1, data_augment=False, return_T2=False, return_segs=False,return_seg3=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """

    while True:
        idxes = np.random.randint(len(vol_names), size=batch_size)
    
        X_data = []
        X_data2 = []
        X_seg = []
        X_seg3 = []
        for idx in idxes:
            X = load_volfile(vol_names[idx], np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            if data_augment: 
                ph_moving_affine=helper.random_transform_generator(batch_size,0.1)
#                ph_moving_affine=helper.random_transform_generator(1)
                X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
            X_data.append(X)
            
            if return_T2:
                X = load_volfile(vol_names[idx].replace('T1_', 'T2_'), np_var=np_var)
                X = X[np.newaxis, ..., np.newaxis]
                if data_augment: 
                    X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
                X_data2.append(X)
            # also return segmentations
            if return_segs:
                segname = vol_names[idx].replace('T1_', 'matter_')
                segname = segname.replace('-denoise-r-h', '')
                X = load_volfile(segname, np_var=np_var)
                X = X[np.newaxis, ..., np.newaxis]
                if data_augment: 
                    X = util1.warp_image_affine(X, ph_moving_affine).astype('float32')  # data augmentation
                X_seg.append(X)
            if return_seg3:
                segname = vol_names[idx].replace('T1_', 'seg3_')
                segname = segname.replace('-denoise-r-h', '')
                X = load_volfile(segname, np_var=np_var)
                #X = X[np.newaxis, ..., np.newaxis]
                X = X[np.newaxis, ...]
                if data_augment: 
                    X0,X1,X2= np.split(X,3,axis=4)
                    X0 = util1.warp_image_affine(X0, ph_moving_affine).astype('float32')  # data augmentation
                    X1 = util1.warp_image_affine(X1, ph_moving_affine).astype('float32')  # data augmentation
                    X2 = util1.warp_image_affine(X2, ph_moving_affine).astype('float32')  # data augmentation
                    X = np.concatenate((X0,X1,X2),axis=4)
                X_seg3.append(X)
        if batch_size > 1:
            return_vals = [np.concatenate(X_data, 0)]
        else:
            return_vals = [X_data[0]]
        if return_T2:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_data2, 0))
            else:
                return_vals.append(X_data2[0])
        if return_segs:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_seg, 0))
            else:
                return_vals.append(X_seg[0])
        if return_seg3:
            if batch_size > 1:
                return_vals.append(np.concatenate(X_seg3, 0))
            else:
                return_vals.append(X_seg3[0])
    
        yield tuple(return_vals)
    
def example_gen1(vol_names, batch_size=1, data_augment=False, return_T2=False, return_segs=False, seg_dir=None, np_var='vol_data'):
    """
    generate examples

    Parameters:
        vol_names: a list or tuple of filenames
        batch_size: the size of the batch (default: 1)

        The following are fairly specific to our data structure, please change to your own
        return_segs: logical on whether to return segmentations
        seg_dir: the segmentations directory.
        np_var: specify the name of the variable in numpy files, if your data is stored in 
            npz files. default to 'vol_data'
    """

#    while True:
    idxes = np.random.randint(len(vol_names), size=batch_size)

    X_data = []
    X_data2 = []
    X_seg = []
    for idx in idxes:
        X = load_volfile(vol_names[idx], np_var=np_var)
        X = X[np.newaxis, ..., np.newaxis]
        if data_augment: 
            ph_moving_affine=helper.random_transform_generator(batch_size,0.1)
            X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
        X_data.append(X)
        
        if return_T2:
            X = load_volfile(vol_names[idx].replace('T1_', 'T2_'), np_var=np_var)
            X = X[np.newaxis, ..., np.newaxis]
            if data_augment: 
                X = (util1.warp_image_affine(X, ph_moving_affine)).astype('float32')  # data augmentation
            X_data2.append(X)
        # also return segmentations
        if return_segs:
            segname = vol_names[idx].replace('T1_', 'seg3_')
            segname = segname.replace('-denoise-r-h', '')
            X = load_volfile(segname, np_var=np_var)
            #X = X[np.newaxis, ..., np.newaxis]
            X = X[np.newaxis, ...]
            if data_augment: 
                X0,X1,X2= np.split(X,3,axis=4)
                X0 = util1.warp_image_affine(X0, ph_moving_affine).astype('float32')  # data augmentation
                X1 = util1.warp_image_affine(X1, ph_moving_affine).astype('float32')  # data augmentation
                X2 = util1.warp_image_affine(X2, ph_moving_affine).astype('float32')  # data augmentation
                X = np.concatenate((X0,X1,X2),axis=4)
            X_seg.append(X)
    if batch_size > 1:
        return_vals = [np.concatenate(X_data, 0)]
    else:
        return_vals = [X_data[0]]
    if return_T2:
        if batch_size > 1:
            return_vals.append(np.concatenate(X_data2, 0))
        else:
            return_vals.append(X_data2[0])
    if return_segs:
        if batch_size > 1:
            return_vals.append(np.concatenate(X_seg, 0))
        else:
            return_vals.append(X_seg[0])

    return tuple(return_vals)
#    yield tuple(return_vals)


def load_example_by_name(vol_name, seg_name, seg3=False, np_var='vol_data'):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_seg = load_volfile(seg_name, np_var)
    if seg3:
        X_seg = X_seg[np.newaxis, ...]
    else:
        X_seg = X_seg[np.newaxis, ..., np.newaxis]
    return_vals.append(X_seg)

    return tuple(return_vals)

def load_example_t1t2_by_name(vol_name, seg_name, np_var='vol_data'):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_2 = load_volfile(vol_name.replace('T1_','T2_'), np_var)
    X_2 = X_2[np.newaxis, ..., np.newaxis]

    return_vals.append(X_2)

    X_seg = load_volfile(seg_name, np_var)
    X_seg = X_seg[np.newaxis, ..., np.newaxis]

    return_vals.append(X_seg)

    return tuple(return_vals)
def load_example_t1t2seg3_by_name(vol_name, seg_name, np_var='vol_data'):
    """
    load a specific volume and segmentation

    np_var: specify the name of the variable in numpy files, if your data is stored in 
        npz files. default to 'vol_data'
    """
    X = load_volfile(vol_name, np_var)
    X = X[np.newaxis, ..., np.newaxis]

    return_vals = [X]

    X_2 = load_volfile(vol_name.replace('T1_','T2_'), np_var)
    X_2 = X_2[np.newaxis, ..., np.newaxis]

    return_vals.append(X_2)

    X_seg = load_volfile(seg_name, np_var)
    X_seg = X_seg[np.newaxis, ...]

    return_vals.append(X_seg)

    return tuple(return_vals)

def load_volfile(datafile, np_var='vol_data'):
    """
    load volume file
    formats: nii, nii.gz, mgz, npz
    if it's a npz (compressed numpy), variable names innp_var (default: 'vol_data')
    """
    assert datafile.endswith(('.nii', '.nii.gz', '.mgz', '.npz')), 'Unknown data file'

    if datafile.endswith(('.nii', '.nii.gz', '.mgz')):
        # import nibabel
#        if 'nibabel' not in sys.modules:
#            try :
#                import nibabel as nib  
#            except:
#                print('Failed to import nibabel. need nibabel library for these data file types.')

        X = nib.load(datafile).get_data()
#        X = (np.array(X)).astype('float32')
        X = X.astype('float32')
#        X = X[...,0]
        
    else: # npz
        if np_var is None:
            np_var = 'vol_data'
        X = np.load(datafile)[np_var]

    return X
