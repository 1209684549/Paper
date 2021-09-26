import sys

# third party
import numpy as np
import keras.backend as K
from keras.models import Model
import keras.layers as KL
from keras.layers import Layer
from keras.layers import Conv3D, Activation, Input, UpSampling3D, Add,Dropout,BatchNormalization,MaxPooling3D,concatenate,add,Multiply, multiply
from keras.layers import LeakyReLU, Reshape, Lambda
from keras.initializers import RandomNormal
import keras.initializers
import tensorflow as tf

# import neuron layers, which will be useful for Transforming.
sys.path.append('../ext')
sys.path.append('../ext/neuron/neuron')
sys.path.append('../ext/pynd-lib')
sys.path.append('../ext/pytools')
#sys.path.append('../ext/pytools-lib')
import neuron.layer as nrn_layers
import neuron.models as nrn_models
import neuron.utils as nrn_utils
#import labelreg.losses as labelloss
# other vm functions
import losses




def unet_core(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-2]])
    x = conv_block(x, dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-3]])
    x = conv_block(x, dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, x_enc[-4]])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        x = upsample_layer()(x)
        x = concatenate([x, x_enc[0]])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src, tgt], outputs=[x])


def attention_unet_core_new_seg(vol_size, enc_nf, dec_nf, full_size=True, src=None, tgt=None, seg=None,src_feats=1, tgt_feats=1):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.
    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)

    # inputs
    if src is None:
        src = Input(shape=[*vol_size, src_feats])
    if tgt is None:
        tgt = Input(shape=[*vol_size, tgt_feats])
    if seg is None:
        seg = Input(shape=[*vol_size, tgt_feats])
    x_in = concatenate([src, tgt])
    

    # down-sample path (encoder)
    x_enc = [x_in]
    for i in range(len(enc_nf)):
        x_enc.append(conv_block(x_enc[-1], enc_nf[i], 2))

    # up-sample path (decoder)
    x = conv_block(x_enc[-1], dec_nf[0])
    enc_att = Attention_block(x_enc[-2],x,dec_nf[0])
    x = upsample_layer()(x)
    x = concatenate([x, enc_att])
    x = conv_block(x, dec_nf[1])
    enc_att = Attention_block(x_enc[-3],x,dec_nf[1])
    x = upsample_layer()(x)
    x = concatenate([x, enc_att])
    x = conv_block(x, dec_nf[2])
    enc_att = Attention_block(x_enc[-4],x,dec_nf[2])
    x = upsample_layer()(x)
    x = concatenate([x, enc_att])
    x = conv_block(x, dec_nf[3])
    x = conv_block(x, dec_nf[4])
    
    # only upsampleto full dim if full_size
    # here we explore architectures where we essentially work with flow fields 
    # that are 1/2 size 
    if full_size:
        enc_att = Attention_block(x_enc[0],x,dec_nf[4])
        x = upsample_layer()(x)
        x = concatenate([x, enc_att])
        x = conv_block(x, dec_nf[5])

    # optional convolution at output resolution (used in voxelmorph-2)
    if len(dec_nf) == 7:
        x = conv_block(x, dec_nf[6])

    return Model(inputs=[src,tgt,seg], outputs=[x,seg])



def cvpr2018_attnet_seg(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
#    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
#    unet_model = attention_unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    unet_model = attention_unet_core_new_seg(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src, tgt,seg] = unet_model.inputs
    x,seg = unet_model.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x)

    # warp the source with the flow
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    y_seg = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([seg, flow])
    # prepare model
    model = Model(inputs=[src, tgt,seg], outputs=[y,y_seg,flow])
    return model






def auto_context_one_attunet_seg(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):

    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims


    unet_model1 = attention_unet_core_new_seg(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src,tgt,seg] = unet_model1.inputs
    x1,seg1 = unet_model1.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow1 = Conv(ndims, kernel_size=3, padding='same', name='flow1',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x1)
    # warp the source with the flow
    y1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow1])
    y_seg1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([seg, flow1])


    x2,seg2 = attention_unet_core_new_seg(vol_size, enc_nf, dec_nf, full_size=full_size)([y1, tgt,seg1])
    flow2 = Conv(ndims, kernel_size=3, padding='same', name='flow2',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x2)
    # warp the source with the flow
    y2 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([y1, flow2])


    flow = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([flow1, flow2])
    flow = add([flow2, flow])


    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
    y_seg2 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([seg2, flow])
	
    # prepare model
    model = Model(inputs=[src, tgt,seg], outputs=[y1,y_seg1,flow1,y2,flow2,y,y_seg2,flow])
    return model
def auto_context_one_3unet(vol_size, enc_nf, dec_nf, full_size=True, indexing='ij'):
    """
    unet architecture for voxelmorph models presented in the CVPR 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6 (like voxelmorph-1) or 1x7 (voxelmorph-2)
    :return: the keras model
    """
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get the core model
    unet_model1 = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)
    [src, tgt] = unet_model1.inputs
    x1 = unet_model1.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow1 = Conv(ndims, kernel_size=3, padding='same', name='flow1',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x1)
    # warp the source with the flow
    y1 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow1])

    nf_dec = [32, 32, 32, 32, 8, 8]
    x2 = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)([y1, tgt])
#    [y1, tgt2] = unet_model2.inputs
#    x2 = unet_model2.output
    # transform the results into a flow field.
    flow2 = Conv(ndims, kernel_size=3, padding='same', name='flow2',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x2)
    # warp the source with the flow
    y2 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([y1, flow2])
    x3 = unet_core(vol_size, enc_nf, nf_dec, full_size=full_size)([y2, tgt])
#    x3 = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size)([y2, tgt])
#    [y1, tgt2] = unet_model2.inputs
#    x2 = unet_model2.output
    # transform the results into a flow field.
    flow3 = Conv(ndims, kernel_size=3, padding='same', name='flow3',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x3)
    # warp the source with the flow
    y3 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([y2, flow3])

#    flow = nrn_utils.compose(flow1, flow2, indexing='ij')
    flow12 = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([flow1, flow2])
    flow12 = add([flow2, flow12])
    flow = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([flow12, flow3])
    flow = add([flow3, flow])
#    x3 = concatenate([x1, x2])
#    x3 = concatenate([flow1,x1,x2])
    # transform the results into a flow field.
#    flow = Conv(ndims, kernel_size=3, padding='same', name='flow',
#                  kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5))(x3)
    # warp the source with the flow
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing)([src, flow])
	
    # prepare model
    model = Model(inputs=[src, tgt], outputs=[y1,y2,y3,y,flow])
    return model





def nn_trf(vol_size, ndim = 1,indexing='xy',interp_method='nearest'):
    """
    Simple transform model for nearest-neighbor based transformation
    Note: this is essentially a wrapper for the neuron.utils.transform(..., interp_method='nearest')
    """
    ndims = len(vol_size)

    # nn warp model
    subj_input = Input((*vol_size,ndim), name='subj_input')
    trf_input = Input((*vol_size, ndims) , name='trf_input')

    # note the nearest neighbour interpolation method
    # note xy indexing because Guha's original code switched x and y dimensions
    nn_output = nrn_layers.SpatialTransformer(interp_method, indexing=indexing)
    nn_spatial_output = nn_output([subj_input, trf_input])
    return keras.models.Model([subj_input, trf_input], nn_spatial_output)


def cvpr2018_net_probatlas(vol_size, enc_nf, dec_nf, nb_labels,
                           diffeomorphic=True,
                           full_size=True,
                           indexing='ij',
                           init_mu=None,
                           init_sigma=None,
                           stat_post_warp=False,  # compute statistics post warp?
                           network_stat_weight=0.001,
                           warp_method='WARP',
                           stat_nb_feats=16):
    """
    Network to do unsupervised segmentation with probabilistic atlas
    (Dalca et al., submitted to MICCAI 2019)
    """
    # print(warp_method)
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    weaknorm = RandomNormal(mean=0.0, stddev=1e-5)

    # get the core model
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size, tgt_feats=nb_labels)
    [src_img, src_atl] = unet_model.inputs
    x = unet_model.output

    # transform the results into a flow field.
    Conv = getattr(KL, 'Conv%dD' % ndims)
    flow1 = Conv(ndims, kernel_size=3, padding='same', name='flow', kernel_initializer=weaknorm)(x)
    if diffeomorphic:
        flow2 = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=8)(flow1)
    else:
        flow2 = flow1
    if full_size:
        flow = flow2
    else:
        flow = trf_resize(flow2, 1/2, name='diffflow')

    # warp atlas
    if warp_method == 'WARP':
        warped_atlas = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_atlas')([src_atl, flow])
    else:
        warped_atlas = src_atl

    if stat_post_warp:
        assert warp_method == 'WARP', "if computing stat post warp, must do warp... :) set warp_method to 'WARP' or stat_post_warp to False?"

        # combine warped atlas and warpedimage and output mu and log_sigma_squared
        combined = concatenate([warped_atlas, src_img])
    else:
        combined = unet_model.layers[-2].output

    conv1 = conv_block(combined, stat_nb_feats)
    conv2 = conv_block(conv1, nb_labels)
    stat_mu_vol = Conv(nb_labels, kernel_size=3, name='mu_vol',
                    kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv2)
    stat_mu = keras.layers.GlobalMaxPooling3D()(stat_mu_vol)
    stat_logssq_vol = Conv(nb_labels, kernel_size=3, name='logsigmasq_vol',
                        kernel_initializer=weaknorm, bias_initializer=weaknorm)(conv2)
    stat_logssq = keras.layers.GlobalMaxPooling3D()(stat_logssq_vol)

    # combine mu with initializtion
    if init_mu is not None: 
        init_mu = np.array(init_mu)
        stat_mu = Lambda(lambda x: network_stat_weight * x + init_mu, name='comb_mu')(stat_mu)
    
    # combine sigma with initializtion
    if init_sigma is not None: 
        init_logsigmasq = np.array([2*np.log(f) for f in init_sigma])
        stat_logssq = Lambda(lambda x: network_stat_weight * x + init_logsigmasq, name='comb_sigma')(stat_logssq)

    # unnorm log-lik
    def unnorm_loglike(I, mu, logsigmasq, uselog=True):
        P = tf.distributions.Normal(mu, K.exp(logsigmasq/2))
        if uselog:
            return P.log_prob(I)
        else:
            return P.prob(I)

    uloglhood = KL.Lambda(lambda x:unnorm_loglike(*x), name='unsup_likelihood')([src_img, stat_mu, stat_logssq])

    # compute data loss as a layer, because it's a bit easier than outputting a ton of things, etc.
    # def logsum(ll, atl):
    #     pdf = ll * atl
    #     return tf.log(tf.reduce_sum(pdf, -1, keepdims=True) + K.epsilon())

    def logsum_safe(prob_ll, atl):
        """
        safe computation using the log sum exp trick
        e.g. https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
        where x = logpdf

        note does not normalize p 
        """
        logpdf = prob_ll + K.log(atl + K.epsilon())
        alpha = tf.reduce_max(logpdf, -1, keepdims=True)
        return alpha + tf.log(tf.reduce_sum(K.exp(logpdf-alpha), -1, keepdims=True) + K.epsilon())

    loss_vol = Lambda(lambda x: logsum_safe(*x))([uloglhood, warped_atlas])

    return Model(inputs=[src_img, src_atl], outputs=[loss_vol, flow])



########################################################
# Atlas creation functions
########################################################



def diff_net(vol_size, enc_nf, dec_nf, int_steps=7, src_feats=1,
             indexing='ij', bidir=False, ret_flows=False, full_size=False,
             vel_resize=1/2, src=None, tgt=None):
    """
    diffeomorphic net, similar to miccai2018, but no sampling.

    architecture for probabilistic diffeomoprhic VoxelMorph presented in the MICCAI 2018 paper. 
    You may need to modify this code (e.g., number of layers) to suit your project needs.

    The stationary velocity field operates in a space (0.5)^3 of vol_size for computational reasons.

    :param vol_size: volume size. e.g. (256, 256, 256)
    :param enc_nf: list of encoder filters. right now it needs to be 1x4.
           e.g. [16,32,32,32]
    :param dec_nf: list of decoder filters. right now it must be 1x6, see unet function.
    :param use_miccai_int: whether to use the manual miccai implementation of scaling and squaring integration
            note that the 'velocity' field outputted in that case was 
            since then we've updated the code to be part of a flexible layer. see neuron.layers.VecInt
            **This param will be phased out (set to False behavior)**
    :param int_steps: the number of integration steps
    :param indexing: xy or ij indexing. we recommend ij indexing if training from scratch. 
            miccai 2018 runs were done with xy indexing.
            **This param will be phased out (set to 'ij' behavior)**
    :return: the keras model
    """    
    ndims = len(vol_size)
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    # get unet
    unet_model = unet_core(vol_size, enc_nf, dec_nf, full_size=full_size, src=src, tgt=tgt, src_feats=src_feats)
    [src, tgt] = unet_model.inputs

    # velocity sample
    # unet_model.layers[-1].name = 'vel'
    # vel = unet_model.output
    x_out = unet_model.outputs[-1]

    # velocity mean and logsigma layers
    Conv = getattr(KL, 'Conv%dD' % ndims)
    vel = Conv(ndims, kernel_size=3, padding='same',
                       kernel_initializer=RandomNormal(mean=0.0, stddev=1e-5), name='flow')(x_out)

    if full_size and vel_resize != 1:
        vel = trf_resize(vel, 1.0/vel_resize, name='flow-resize')    

    # new implementation in neuron is cleaner.
    flow = nrn_layers.VecInt(method='ss', name='flow-int', int_steps=int_steps)(vel)
    if bidir:
        # rev_z_sample = Lambda(lambda x: -x)(z_sample)
        neg_vel = Negate()(vel)
        neg_flow = nrn_layers.VecInt(method='ss', name='neg_flow-int', int_steps=int_steps)(neg_vel)

    # get up to final resolution
    flow = trf_resize(flow, vel_resize, name='diffflow')
    if bidir:
        neg_flow = trf_resize(neg_flow, vel_resize, name='neg_diffflow')

    # transform
    y = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_src')([src, flow])
    if bidir:
        y_tgt = nrn_layers.SpatialTransformer(interp_method='linear', indexing=indexing, name='warped_tgt')([tgt, neg_flow])

    # prepare outputs and losses
    outputs = [y, vel]
    if bidir:
        outputs = [y, y_tgt, vel]

    model = Model(inputs=[src, tgt], outputs=outputs)

    if ret_flows:
        outputs += [model.get_layer('diffflow').output, model.get_layer('neg_diffflow').output]
        return Model(inputs=[src, tgt], outputs=outputs)
    else: 
        return model


def atl_img_model(vol_shape, mult=1.0, src=None, atl_layer_name='img_params'):
    """
    atlas model with flow representation
    idea: starting with some (probably rough) atlas (like a ball or average shape),
    the output atlas is this input ball plus a 
    """

    # get a new layer (std)
    if src is None:
        src = Input(shape=[*vol_shape, 1], name='input_atlas')

    # get the velocity field
    v_layer = LocalParamWithInput(shape=[*vol_shape, 1],
                                  mult=mult,
                                  name=atl_layer_name,
                                  my_initializer=RandomNormal(mean=0.0, stddev=1e-7))
    v = v_layer(src)  # this is so memory-wasteful...

    return keras.models.Model(src, v)





def cond_img_atlas_diff_model(vol_shape, nf_enc, nf_dec,
                                atl_mult=1.0,
                                bidir=True,
                                smooth_pen_layer='diffflow',
                                vel_resize=1/2,
                                int_steps=5,
                                nb_conv_features=32,
                                cond_im_input_shape=[10,12,14,1],
                                cond_nb_levels=5,
                                cond_conv_size=[3,3,3],
                                use_stack=True,
                                do_mean_layer=True,
                                pheno_input_shape=[1],
                                atlas_feats=1,
                                name='cond_model',
                                mean_cap=100,
                                templcondsi=False,
                                templcondsi_init=None,
                                full_size=False,
                                ret_vm=False,
                                extra_conv_layers=0,
                                **kwargs):
            
    # conv layer class
    Conv = getattr(KL, 'Conv%dD' % len(vol_shape))

    # vm model. inputs: "atlas" (we will replace this) and 
    mn = diff_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, bidir=bidir, src_feats=atlas_feats,
                  full_size=full_size, vel_resize=vel_resize, ret_flows=(not use_stack), **kwargs)
    
    # pre-warp model (atlas model)
    pheno_input = KL.Input(pheno_input_shape, name='pheno_input')
    dense_tensor = KL.Dense(np.prod(cond_im_input_shape), activation='elu')(pheno_input)
    reshape_tensor = KL.Reshape(cond_im_input_shape)(dense_tensor)
    pheno_init_model = keras.models.Model(pheno_input, reshape_tensor)
    pheno_tmp_model = nrn_models.conv_dec(nb_conv_features, cond_im_input_shape, cond_nb_levels, cond_conv_size,
                             nb_labels=nb_conv_features, final_pred_activation='linear',
                             input_model=pheno_init_model, name='atlasmodel')
    last_tensor = pheno_tmp_model.output
    for i in range(extra_conv_layers):
        last_tensor = Conv(nb_conv_features, kernel_size=cond_conv_size, padding='same', name='atlas_ec_%d' % i)(last_tensor)
    pout = Conv(atlas_feats, kernel_size=3, padding='same', name='atlasmodel_c',
                 kernel_initializer=RandomNormal(mean=0.0, stddev=1e-7),
                 bias_initializer=RandomNormal(mean=0.0, stddev=1e-7))(last_tensor)
    atlas_input = KL.Input([*vol_shape, atlas_feats], name='atlas_input')
    if not templcondsi:
        atlas_tensor = KL.Add(name='atlas')([atlas_input, pout])
    else:
        atlas_tensor = KL.Add(name='atlas_tmp')([atlas_input, pout])

        # change first channel to be result from seg with another add layer
        tmp_layer = KL.Lambda(lambda x: K.softmax(x[...,1:]))(atlas_tensor)  # this is just tmp. Do not use me.
        cl = Conv(1, kernel_size=1, padding='same', use_bias=False, name='atlas_gen', kernel_initializer=RandomNormal(mean=0, stddev=1e-5))
        ximg = cl(tmp_layer)
        if templcondsi_init is not None:
            w = cl.get_weights()
            w[0] = templcondsi_init.reshape(w[0].shape)
            cl.set_weights(w)
        atlas_tensor = KL.Lambda(lambda x: K.concatenate([x[0], x[1][...,1:]]), name='atlas')([ximg, atlas_tensor]) 

    pheno_model = keras.models.Model([pheno_tmp_model.input, atlas_input], atlas_tensor)

    # stack models
    inputs = pheno_model.inputs + [mn.inputs[1]]

    if use_stack:
        sm = nrn_utils.stack_models([pheno_model, mn], [[0]])
        neg_diffflow_out = sm.get_layer('neg_diffflow').get_output_at(-1)
        diffflow_out = mn.get_layer(smooth_pen_layer).get_output_at(-1)
        warped_src = sm.get_layer('warped_src').get_output_at(-1)
        warped_tgt = sm.get_layer('warped_tgt').get_output_at(-1)

    else:
        assert bidir
        assert smooth_pen_layer == 'diffflow'
        warped_src, warped_tgt, _, diffflow_out, neg_diffflow_out = mn(pheno_model.outputs + [mn.inputs[1]])
        sm = keras.models.Model(inputs, [warped_src, warped_tgt])
        
    if do_mean_layer:
        mean_layer = nrn_layers.MeanStream(name='mean_stream', cap=mean_cap)(neg_diffflow_out)
        outputs = [warped_src, warped_tgt, mean_layer, diffflow_out]
    else:
        outputs = [warped_src, warped_tgt, diffflow_out]


    model = keras.models.Model(inputs, outputs, name=name)
    if ret_vm:
        return model, mn
    else:
        return model





def img_atlas_diff_model(vol_shape, nf_enc, nf_dec,
                        atl_mult=1.0,
                        bidir=True,
                        smooth_pen_layer='diffflow',
                        atl_int_steps=3,
                        vel_resize=1/2,
                        int_steps=3,
                        mean_cap=100,
                        atl_layer_name='atlas',
                        **kwargs):
    
    

    # vm model
    mn = diff_net(vol_shape, nf_enc, nf_dec, int_steps=int_steps, bidir=bidir, 
                        vel_resize=vel_resize, **kwargs)
    
    # pre-warp model (atlas model)
    pw = atl_img_model(vol_shape, mult=atl_mult, src=mn.inputs[0], atl_layer_name=atl_layer_name) # Wait I'm confused....

    # stack models
    sm = nrn_utils.stack_models([pw, mn], [[0]])
    # note: sm.outputs might be out of order now

    # TODO: I'm not sure the mean layer is the right direction
    mean_layer = nrn_layers.MeanStream(name='mean_stream', cap=mean_cap)(sm.get_layer('neg_diffflow').get_output_at(-1))

    outputs = [sm.get_layer('warped_src').get_output_at(-1),
               sm.get_layer('warped_tgt').get_output_at(-1),
               mean_layer,
               mn.get_layer(smooth_pen_layer).get_output_at(-1)]

    model = keras.models.Model(mn.inputs, outputs)
    return model




########################################################
# Helper functions
########################################################

def conv_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

def BatchActivate(x):
    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
    x = LeakyReLU(0.2)(x)
    return x

def res_conv_block(x_in, nf, kernel_size=3, strides=1, activation=True):
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x    = Conv(nf, kernel_size, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    if activation == True:
        x = BatchActivate(x)
    return x

def residual_block(blockInput, nf, strides=1, batch_activate=False):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    x = BatchActivate(blockInput)
    x = res_conv_block(x, nf)
    x = res_conv_block(x, nf, activation=False)
    x = Add()([x, blockInput])
    if batch_activate:
        x = BatchActivate(x)
    return x

def dense_block(x_in, nf, strides=1):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(x_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims

    Conv = getattr(KL, 'Conv%dD' % ndims)
    x_out = Conv(nf, kernel_size=3, padding='same',
                 kernel_initializer='he_normal', strides=strides)(x_in)
    x_out = LeakyReLU(0.2)(x_out)
    return x_out

def sample(args):
    """
    sample from a normal distribution
    """
    mu = args[0]
    log_sigma = args[1]
    noise = tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    z = mu + tf.exp(log_sigma/2.0) * noise
    return z


def trf_resize(trf, vel_resize, name='flowResize'):
    if vel_resize > 1:
        trf = nrn_layers.Resize(1/vel_resize, name=name+'_tmp')(trf)
        return Rescale(1 / vel_resize, name=name)(trf)

    else: # multiply first to save memory (multiply in smaller space)
        trf = Rescale(1 / vel_resize, name=name+'_tmp')(trf)
        return  nrn_layers.Resize(1/vel_resize, name=name)(trf)


class Sample(Layer):
    """ 
    Keras Layer: Gaussian sample from [mu, sigma]
    """

    def __init__(self, **kwargs):
        super(Sample, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Sample, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class Negate(Layer):
    """ 
    Keras Layer: negative of the input
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape

class Rescale(Layer):
    """ 
    Keras layer: rescale data by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(Rescale, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Rescale, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize 

    def compute_output_shape(self, input_shape):
        return input_shape

class RescaleDouble(Rescale):
    def __init__(self, **kwargs):
        self.resize = 2
        super(RescaleDouble, self).__init__(self.resize, **kwargs)
'''
class ResizeDouble(nrn_layers.Resize):
    def __init__(self, **kwargs):
        self.zoom_factor = 2
        super(ResizeDouble, self).__init__(self.zoom_factor, **kwargs)

'''
class LocalParamWithInput(Layer):
    """ 
    The neuron.layers.LocalParam has an issue where _keras_shape gets lost upon calling get_output :(
        tried using call() but this requires an input (or i don't know how to fix it)
        the fix was that after the return, for every time that tensor would be used i would need to do something like
        new_vec._keras_shape = old_vec._keras_shape

        which messed up the code. Instead, we'll do this quick version where we need an input, but we'll ignore it.

        this doesn't have the _keras_shape issue since we built on the input and use call()
    """

    def __init__(self, shape, my_initializer='RandomNormal', mult=1.0, **kwargs):
        self.shape=shape
        self.initializer = my_initializer
        self.biasmult = mult
        super(LocalParamWithInput, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(name='kernel', 
                                      shape=self.shape,  # input_shape[1:]
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalParamWithInput, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        # want the x variable for it's keras properties and the batch.
        b = 0*K.batch_flatten(x)[:,0:1] + 1
        params = K.expand_dims(K.flatten(self.kernel * self.biasmult), 0)
        z = K.reshape(K.dot(b, params), [-1, *self.shape])
        return z

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.shape)

class MergeInputs3D(Layer):
    """ 
    The MergeInputs3D Layer merge two 3D inputs with Inputs1*alpha+Inputs2*beta, where alpha+beta=1.
    keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
    keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))
    """
    def __init__(self, output_dim=1, my_initializer='RandomUniform', **kwargs):
#        self.shape=shape
        self.initializer = my_initializer
        self.output_dim = output_dim
        super(MergeInputs3D, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        if len(input_shape) > 2:
            raise Exception('must be called on a list of length 2.')
        
        # set up number of dimensions
#        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
#        shape_kernel = [self.output_dim]
#        shape_kernel.append()
#        shape_kernel1 = tuple(shape_kernel)
        shape_a = input_shape[0][1:]#input_shape[0][1:-1]
#        shape_b = input_shape[1][1:]
#        shape_a, shape_b = input_shape
#        assert shape_a == shape_b
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=shape_a,
                                      initializer= RandomNormal(mean=0.5,stddev=0.001),
#                                      initializer=self.initializer,#'uniform',
#                                      constraint = min_max_norm,
                                      trainable=True)
        super(MergeInputs3D, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        # check shapes
        assert isinstance(inputs, list)
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol1 = inputs[0]
        vol2 = inputs[1]
        batch_size = tf.shape(vol1)[0]
#        height = tf.shape(vol1)[1]
#        width = tf.shape(vol1)[2]
#        depth = tf.shape(vol1)[3]
#        channels = tf.shape(vol1)[4]
#        vol1, vol2 = inputs

        # necessary for multi_gpu models...
#        vol1 = K.reshape(vol1, [-1, *self.inshape[0][1:]])
#        vol2 = K.reshape(vol2, [-1, *self.inshape[1][1:]])
        
        alpha = self.kernel# + 0.5
        alpha = tf.expand_dims(alpha, 0)
        alpha = tf.tile(alpha, [batch_size, 1, 1, 1, 1])
        beta = 1- alpha
        return vol1*alpha + vol2*beta

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
#        shape_a = input_shape[0][1:-1]
#        shape_b = input_shape[1][1:]
#        shape_a, shape_b = input_shape
#        assert shape_a == shape_b
        return input_shape[0]
    
class NCCLayer(Layer):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=None,channels=1,alpha=0.5, eps=1e-5, **kwargs):
        self.win = win
        self.channels = channels
        self.alpha = alpha
        self.eps = eps
        super(NCCLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape

        # confirm built
        self.built = True


    def call(self, inputs):
        # check shapes
        assert isinstance(inputs, list)
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        I = inputs[0]
        J = inputs[1]
#        batch_size = tf.shape(vol1)[0]

        # get dimension of volume
        # assumes I, J are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(I.get_shape().as_list()) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        if self.win is None:
            self.win = [9] * ndims

        # get convolution function
        conv_fn = getattr(tf.nn, 'conv%dd' % ndims)

        # compute CC squares
        I2 = I*I
        J2 = J*J
        IJ = I*J
        # compute filters
#        sum_filt = tf.ones([*self.win,self.channels, self.channels])
        sum_filt = tf.ones([*self.win,self.channels, self.channels])
        strides = 1
        if ndims > 1:
            strides = [1] * (ndims + 2)
        padding = 'SAME'

        # compute local sums via convolution
        I_sum = conv_fn(I, sum_filt, strides, padding)
        J_sum = conv_fn(J, sum_filt, strides, padding)
        I2_sum = conv_fn(I2, sum_filt, strides, padding)
        J2_sum = conv_fn(J2, sum_filt, strides, padding)
        IJ_sum = conv_fn(IJ, sum_filt, strides, padding)

        # compute cross correlation
        win_size = np.prod(self.win)
        u_I = I_sum/win_size
        u_J = J_sum/win_size

        cross = IJ_sum - u_J*I_sum - u_I*J_sum + u_I*u_J*win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I*u_I*win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J*u_J*win_size

        cc = cross*cross / (I_var*J_var + self.eps)

        # return negative cc.
#        return cc
        return 1 - tf.reduce_mean(cc)
    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
#        shape_a = input_shape[0][1:-1]
#        shape_b = input_shape[1][1:]
#        shape_a, shape_b = input_shape
#        assert shape_a == shape_b
        return (1,1,1)
class MyTile(Layer):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, channels=1,**kwargs):
        self.channels = channels
        super(MyTile, self).__init__(**kwargs)
        
    def build(self, input_shape):
        # set up number of dimensions
        self.inputshape = input_shape
        # confirm built
        self.built = True
    def call(self, inputs):
        # check shapes
        inputs = tf.tile(inputs, [1, 1, 1, 1, self.channels])
        return inputs

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 5  # only valid for 2D tensors
        shape[-1] *= self.channels
        return tuple(shape)
    
def Attention_block(up_in,down_in,nf):
    """
    specific convolution module including convolution followed by leakyrelu
    """
#    def MyTile(alpha, channels):
#        alpha = tf.tile(alpha, [1, 1, 1, 1, channels])
#        return alpha
#    def MyTile_output_shape(input_shape):
#        shape = list(input_shape)
#        assert len(shape) == 5  # only valid for 2D tensors
#        shape[-1] *= channels
#        return tuple(shape)
    ndims = len(up_in.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
    input_channels = up_in.get_shape().as_list()[-1]
#    batch_size1 = tf.shape(down_in)[0]
#    nf  = tf.min(batch_size0,batch_size1)
    Conv = getattr(KL, 'Conv%dD' % ndims)
    up = Conv(nf, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', strides=2)(up_in)
    down = Conv(nf, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', strides=1)(down_in)

#    x = NCCLayer(channels=nf)([up,down])
    x = Add()([up,down])
    x = Activation('relu')(x)
    x = Conv(1, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', use_bias = True, bias_initializer='zeros',strides=1,activation='sigmoid')(x)
#    x = Activation('sigmoid')(x)
    upsample_layer = getattr(KL, 'UpSampling%dD' % ndims)
    alpha = upsample_layer()(x)
#    alpha = Lambda(MyTile)(alpha, input_channels)
    alpha = MyTile(channels=input_channels)(alpha)
    up_out = Multiply()([alpha,up_in])
    return up_out

def Anatomical_attention_gate(featureMap1,featureMap2):
    """
    specific convolution module including convolution followed by leakyrelu
    """
    ndims = len(featureMap1.get_shape()) - 2
    assert ndims in [1, 2, 3], "ndims should be one of 1, 2, or 3. found: %d" % ndims
#    input_channels = featureMap1.get_shape().as_list()[-1]
#    batch_size1 = tf.shape(down_in)[0]
#    nf  = tf.min(batch_size0,batch_size1)
    featureMap = concatenate([featureMap1, featureMap2])
    Conv = getattr(KL, 'Conv%dD' % ndims)
    tensorweight1 = Conv(1, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', use_bias = True, bias_initializer='zeros',strides=1,activation='sigmoid')(featureMap)
#    tensorweight1 = Activation('relu')(tensorweight1)
    w_featureMap1 = Multiply()([featureMap1,tensorweight1])
    tensorweight2 = Conv(1, kernel_size=1, padding='same',
                 kernel_initializer='he_normal', use_bias = True, bias_initializer='zeros',strides=1,activation='sigmoid')(featureMap)
#    tensorweight2 = Activation('relu')(tensorweight2)
    w_featureMap2 = Multiply()([featureMap2,tensorweight2])
    w_featureMap = Add()([w_featureMap1,w_featureMap2])
    return w_featureMap