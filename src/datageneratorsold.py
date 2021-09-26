import os
import sys
#import minpy.numpy as np
import numpy as np
import nibabel as nib
sys.path.append('../ext')
import labelreg.helpers as helper
import labelreg.utils1 as util1

def load_example_by_name(vol_name, seg_name):

    X = np.load(vol_name)['vol_data']
    X = np.reshape(X, (1,) + X.shape + (1,))

    return_vals = [X]

    X_seg = np.load(seg_name)['vol_data']
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)

    return tuple(return_vals)
def load_example_by_name1_nii(vol_name1, seg_name):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
    ref_affine = X.affine.astype('float32')
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X = np.reshape(X, (1,) + X.shape + (1,))
    return_vals = [X]

    #X_seg = np.load(seg_name)['vol_data']
    X_seg = nib.load(seg_name)
    X_seg = X_seg.get_fdata()
    X_seg = (np.array(X_seg)).astype('float32')
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)
    return_vals.append(ref_affine)
    return tuple(return_vals)
def load_testexample_by_name_nii(vol_namesT1, return_segs=False):
    idx = -1
    while(True):
        idx = (idx+1)%len(vol_namesT1)
#        idx = np.random.randint(len(vol_namesT1))
        img = nib.load(vol_namesT1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
#        ph_moving_affine=helper.random_transform_generator(1,0.00001)
#        X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals = [X1]

        vol_names2 = vol_namesT1[idx].replace('T1_','T2_')
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
#        X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals.append(X2)
        if(return_segs):
            seg_name = vol_namesT1[idx].replace('T1_','matter_')
            X_seg = nib.load(seg_name.replace('-denoise-r-h.nii','.nii'))
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
#            X_seg = (util1.warp_image_affine(X_seg, ph_moving_affine)).astype('float32')  # data augmentation
            return_vals.append(X_seg)
        yield tuple(return_vals)
def load_testexample_by_name1_nii(vol_namesT1, return_segs=False):
    idx = -1
    while(True):
        idx = (idx+1)%len(vol_namesT1)
#        idx = np.random.randint(len(vol_namesT1))
        img = nib.load(vol_namesT1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
#        ph_moving_affine=helper.random_transform_generator(1,0.00001)
#        X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals = [X1]

        if(return_segs):
            seg_name = vol_namesT1[idx].replace('gfc.nii.gz','gfc_fseg.nii.gz')
            X_seg = nib.load(seg_name)
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
#            X_seg = (util1.warp_image_affine(X_seg, ph_moving_affine)).astype('float32')  # data augmentation
            return_vals.append(X_seg)
        yield tuple(return_vals)
def load_testexample_by_name_nii4(vol_names1, return_segs=False, seg_name=None):
    idx = -1
    while(True):
        idx = (idx+1)%len(vol_names1)
        img = nib.load(vol_names1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        X_grad = np.gradient(np.squeeze(X1)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X1_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X1_grad = np.squeeze(np.array(X1_grad).astype('float32')/255)
        X1_grad = np.reshape(X1_grad, (1,) + X1_grad.shape + (1,))
        return_vals = [X1]
#        return_vals_grad = [X1_grad]
        return_vals.append(X1_grad)

        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        X_grad = np.gradient(np.squeeze(X2)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X2_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X2_grad = np.squeeze(np.array(X2_grad).astype('float32')/255)
        X2_grad = np.reshape(X2_grad, (1,) + X2_grad.shape + (1,))
#        return_vals2 = [X2]
#        return_vals2_grad = [X2_grad]
        return_vals.append(X2)
        return_vals.append(X2_grad)
        if(return_segs):
            seg_name = vol_names1[idx].replace('T1_','matter_')
            X_seg = nib.load(seg_name.replace('-denoise-r-h.nii','.nii'))
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals.append(X_seg)
#        yield tuple(return_vals1),tuple(return_vals1_grad),tuple(return_vals2),tuple(return_vals2_grad)        
        yield tuple(return_vals)

def load_example_by_name_nii(vol_name1,vol_name2, seg_name):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
    ref_affine = X.affine.astype('float32')
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X = np.reshape(X, (1,) + X.shape + (1,))
    return_vals = [X]
    
    X2 = nib.load(vol_name2)
    X2 = X2.get_fdata()
    X2 = (np.array(X2)).astype('float32')
    X2 = np.reshape(X2, (1,) + X2.shape + (1,))
    return_vals.append(X2)

    #X_seg = np.load(seg_name)['vol_data']
    X_seg = nib.load(seg_name)
    X_seg = X_seg.get_fdata()
    X_seg = (np.array(X_seg)).astype('float32')
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)
    return_vals.append(ref_affine)
    return tuple(return_vals)
def load_example_by_name_seg3(vol_name1,returnT2=False,returnseg = False,returnseg3 = False, returnhippo = False):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
    ref_affine = X.affine.astype('float32')
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X = np.reshape(X, (1,) + X.shape + (1,))
    return_vals = [X]
    
    if returnT2:
        vol_name2 = vol_name1.replace('T1_','T2_')
        X2 = nib.load(vol_name2)
        X2 = X2.get_fdata()
        X2 = (np.array(X2)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        return_vals.append(X2)

        vol_name1 = vol_name1.replace('-denoise-r-h','')
    if returnseg:
        seg_name = vol_name1.replace('T1_','matter_')        
        #X_seg = np.load(seg_name)['vol_data']
        X_seg = nib.load(seg_name)
        X_seg = X_seg.get_fdata()
        X_seg = (np.array(X_seg)).astype('int16')
        X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
        return_vals.append(X_seg)
    if returnseg3:
        seg_name = vol_name1.replace('T1_','seg3_')
        X_seg = nib.load(seg_name)
        X_seg = X_seg.get_fdata()
        X_seg = (np.array(X_seg)).astype('int16')
        X_seg = np.reshape(X_seg, (1,) + X_seg.shape)
        return_vals.append(X_seg)
    if returnhippo:
        seg_name = vol_name1.replace('T1_','hippo_')
        X_seg = nib.load(seg_name)
        X_seg = X_seg.get_fdata()
        X_seg = (np.array(X_seg)).astype('int16')
        X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
        return_vals.append(X_seg)    
    return_vals.append(ref_affine)
    return tuple(return_vals)
def load_example_by_name2_nii(vol_name1,vol_name2):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
#    ref_affine = X.affine.astype('float32')
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X = np.reshape(X, (1,) + X.shape + (1,))
    return_vals = [X]
    
    X2 = nib.load(vol_name2)
    X2 = X2.get_fdata()
    X2 = (np.array(X2)).astype('float32')
    X2 = np.reshape(X2, (1,) + X2.shape + (1,))
    return_vals.append(X2)

#    #X_seg = np.load(seg_name)['vol_data']
#    X_seg = nib.load(seg_name)
#    X_seg = X_seg.get_fdata()
#    X_seg = (np.array(X_seg)).astype('float32')
#    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
#    return_vals.append(X_seg)
#    return_vals.append(ref_affine)
    return tuple(return_vals)
def load_example_by_namenii(vol_name1):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
#    ref_affine = X.affine.astype('float32')
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X = np.reshape(X, (1,) + X.shape + (1,))
#    return_vals = [X]
    return [X]
def load_example_by_name_nii_grad(vol_name1,vol_name2, seg_name):

    #X = np.load(vol_name)['vol_data']
    X = nib.load(vol_name1)
    X = X.get_fdata()
    X = (np.array(X)).astype('float32')
    X_grad = np.gradient(np.squeeze(X)*255)
    X_grad[0] = pow(X_grad[0],2)
    X_grad[1] = pow(X_grad[1],2)
    X_grad[2] = pow(X_grad[2],2)
    X_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
    X_grad = np.squeeze(np.array(X_grad).astype('float32')/255)
    X = np.reshape(X, (1,) + X.shape + (1,))
    X_grad = np.reshape(X_grad, (1,) + X_grad.shape + (1,))
    return_vals = [X]
    return_vals.append(X_grad)
    
    X2 = nib.load(vol_name2)
    X2 = X2.get_fdata()
    X2 = (np.array(X2)).astype('float32')
    X_grad = np.gradient(np.squeeze(X2)*255)
    X_grad[0] = pow(X_grad[0],2)
    X_grad[1] = pow(X_grad[1],2)
    X_grad[2] = pow(X_grad[2],2)
    X2_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
    X2_grad = np.squeeze(np.array(X2_grad).astype('float32')/255)
    X2 = np.reshape(X2, (1,) + X2.shape + (1,))
    X2_grad = np.reshape(X2_grad, (1,) + X2_grad.shape + (1,))
    return_vals.append(X2)
    return_vals.append(X2_grad)

    #X_seg = np.load(seg_name)['vol_data']
    X_seg = nib.load(seg_name)
    X_seg = X_seg.get_fdata()
    X_seg = (np.array(X_seg)).astype('float32')
    X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
    return_vals.append(X_seg)

    return tuple(return_vals)
def example_gen(vol_names, return_segs=False, seg_dir=None):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names))
        X = np.load(vol_names[idx])['vol_data']
        X = np.reshape(X, (1,) + X.shape + (1,))
        

        return_vals = [X]

        if(return_segs):
            name = os.path.basename(vol_names[idx])
            X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals.append(X_seg)

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'

        yield tuple(return_vals)

def example_gen2(vol_names1, return_segs=False, seg_dir=None):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names1))
        X1 = np.load(vol_names1[idx])['vol_data']
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        return_vals1 = [X1]

        if(return_segs):
            name = os.path.basename(vol_names1[idx])
            X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals1.append(X_seg)

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'
        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        X2 = np.load(vol_names2)['vol_data']
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        #return_vals1.append(X2)
        return_vals2 = [X2]

        yield tuple(return_vals1),tuple(return_vals2)
def example_gen3(vol_names1, return_segs=False, seg_dir=None):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names1))
        img = nib.load(vol_names1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        return_vals1 = [X1]

        if(return_segs):
            name = os.path.basename(vol_names1[idx])
            X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals1.append(X_seg)

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'
        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        #X2 = np.load(vol_names2)['vol_data']
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        #return_vals1.append(X2)
        return_vals2 = [X2]

        yield tuple(return_vals1),tuple(return_vals2)
        
def example_gen4(vol_names1, return_segs=False, seg_dir=None):
    idx = -1
    while(True):
#        idx = (idx+1)%len(vol_names1)
        idx = np.random.randint(len(vol_names1))
        img = nib.load(vol_names1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        X_grad = np.gradient(np.squeeze(X1)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X1_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X1_grad = np.squeeze(np.array(X1_grad).astype('float32')/255)
        X1_grad = np.reshape(X1_grad, (1,) + X1_grad.shape + (1,))
        return_vals = [X1]
        return_vals.append(X1_grad)

        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        X_grad = np.gradient(np.squeeze(X2)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X2_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X2_grad = np.squeeze(np.array(X2_grad).astype('float32')/255)
        X2_grad = np.reshape(X2_grad, (1,) + X2_grad.shape + (1,))
        return_vals.append(X2)
        return_vals.append(X2_grad)

        if(return_segs):
            seg_name = vol_names1[idx].replace('T1_','matter_')
            X_seg = nib.load(seg_name.replace('-denoise-r-h.nii','.nii'))
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals.append(X_seg)
        yield tuple(return_vals)
        
def example_gen4_random(vol_names1, return_segs=False, seg_name=None):
    idx = -1
    while(True):
#        idx = (idx+1)%len(vol_names1)
        idx = np.random.randint(len(vol_names1))
        img = nib.load(vol_names1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        ph_moving_affine=helper.random_transform_generator(1,0.00001)
        X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')  # data augmentation
        X_grad = np.gradient(np.squeeze(X1)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X1_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X1_grad = np.squeeze(np.array(X1_grad).astype('float32')/255)
        X1_grad = np.reshape(X1_grad, (1,) + X1_grad.shape + (1,))
        return_vals = [X1]
#        return_vals_grad = [X1_grad]
        return_vals.append(X1_grad)

        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')  # data augmentation
        X_grad = np.gradient(np.squeeze(X2)*255)
        X_grad[0] = pow(X_grad[0],2)
        X_grad[1] = pow(X_grad[1],2)
        X_grad[2] = pow(X_grad[2],2)
        X2_grad = np.sqrt(X_grad[0]+X_grad[1]+X_grad[2])
        X2_grad = np.squeeze(np.array(X2_grad).astype('float32')/255)
        X2_grad = np.reshape(X2_grad, (1,) + X2_grad.shape + (1,))
#        return_vals2 = [X2]
#        return_vals2_grad = [X2_grad]
        return_vals.append(X2)
        return_vals.append(X2_grad)
        if(return_segs):
            seg_name = vol_names1[idx].replace('T1_','matter_')
            X_seg = nib.load(seg_name.replace('-denoise-r-h.nii','.nii'))
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            X_seg = (util1.warp_image_affine(X_seg, ph_moving_affine)).astype('float32')  # data augmentation
            return_vals.append(X_seg)
#        yield tuple(return_vals1),tuple(return_vals1_grad),tuple(return_vals2),tuple(return_vals2_grad)        
        yield tuple(return_vals)
def example_gen2_random(vol_names1, return_segs=False, seg_name=None):
    idx = -1
    while(True):
#        idx = (idx+1)%len(vol_names1)
        idx = np.random.randint(len(vol_names1))
        img = nib.load(vol_names1[idx])
        img = img.get_fdata()
        X1 = (np.array(img)).astype('float32')
        X1 = np.reshape(X1, (1,) + X1.shape + (1,))
        ph_moving_affine=helper.random_transform_generator(1,0.00001)
        X1 = (util1.warp_image_affine(X1, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals = [X1]

        vol_names2 = vol_names1[idx].replace('T1_','T2_')
        img = nib.load(vol_names2)
        img = img.get_fdata()
        X2 = (np.array(img)).astype('float32')
        X2 = np.reshape(X2, (1,) + X2.shape + (1,))
        X2 = (util1.warp_image_affine(X2, ph_moving_affine)).astype('float32')  # data augmentation
        return_vals.append(X2)
        if(return_segs):
            seg_name = vol_names1[idx].replace('T1_','matter_')
            X_seg = nib.load(seg_name.replace('-denoise-r-h.nii','.nii'))
            X_seg = X_seg.get_fdata()
            X_seg = (np.array(X_seg)).astype('float32')
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            X_seg = (util1.warp_image_affine(X_seg, ph_moving_affine)).astype('float32')  # data augmentation
            return_vals.append(X_seg)
        yield tuple(return_vals)