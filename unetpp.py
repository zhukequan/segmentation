from __future__ import print_function, division, absolute_import, unicode_literals

import keras
from keras.models import Model
from keras import backend as K
from keras.layers import Input, merge, Conv2D, ZeroPadding2D, UpSampling2D, Dense, concatenate, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import BatchNormalization, Dropout, Flatten, Lambda
from keras.layers.advanced_activations import ELU, LeakyReLU
from keras.optimizers import Adam, RMSprop, SGD
from keras.regularizers import l2
from keras.layers.noise import GaussianDropout
import numpy as np
import tensorflow as tf
import os
smooth = 1.
keep_rate = 0.5
act = "relu"

def standard_unit(input_tensor, stage, nb_filter, kernel_size=3):
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_1',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(input_tensor)
    x = tf.nn.dropout(x,keep_prob=keep_rate, name='dp' + stage + '_1')
    #x = Dropout(dropout_rate, name='dp' + stage + '_1')(x)
    x = Conv2D(nb_filter, (kernel_size, kernel_size), activation=act, name='conv' + stage + '_2',
               kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(1e-4))(x)
    #x = Dropout(dropout_rate, name='dp' + stage + '_2')(x)
    x = tf.nn.dropout(x, keep_prob=keep_rate, name='dp' + stage + '_1')

    return x


def pixel_wise_softmax(output_maps ,mask):
    softmax_maps = []
    with tf.name_scope("pixel_wise_softmax"):
        for output_map in output_maps:
            max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
            exponential_map = tf.exp(output_map - max_axis)
            normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
            softmax_maps.append(exponential_map / normalize*mask)
    return softmax_maps

def Nest_Net(img_input, keep_prob = 0.5 ,num_class=1, deep_supervision=True):
    nb_filter = [32, 64, 128, 256, 512]

    # Handle Dimension Ordering for different backends
    global bn_axis
    global keep_rate
    keep_rate = keep_prob
    bn_axis = 3
    # if K.image_dim_ordering() == 'tf':
    #     bn_axis = 3
    #     img_input = Input(shape=(img_rows, img_cols, color_type), name='main_input')
    # else:
    #     bn_axis = 1
    #     img_input = Input(shape=(color_type, img_rows, img_cols), name='main_input')

    conv1_1 = standard_unit(img_input, stage='11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D((2, 2), strides=(2, 2), name='pool1')(conv1_1)

    conv2_1 = standard_unit(pool1, stage='21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2), strides=(2, 2), name='pool2')(conv2_1)

    up1_2 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up12', padding='same')(conv2_1)
    conv1_2 = concatenate([up1_2, conv1_1], name='merge12', axis=bn_axis)
    conv1_2 = standard_unit(conv1_2, stage='12', nb_filter=nb_filter[0])

    conv3_1 = standard_unit(pool2, stage='31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2), strides=(2, 2), name='pool3')(conv3_1)

    up2_2 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up22', padding='same')(conv3_1)
    conv2_2 = concatenate([up2_2, conv2_1], name='merge22', axis=bn_axis)
    conv2_2 = standard_unit(conv2_2, stage='22', nb_filter=nb_filter[1])

    up1_3 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up13', padding='same')(conv2_2)
    conv1_3 = concatenate([up1_3, conv1_1, conv1_2], name='merge13', axis=bn_axis)
    conv1_3 = standard_unit(conv1_3, stage='13', nb_filter=nb_filter[0])

    conv4_1 = standard_unit(pool3, stage='41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2), strides=(2, 2), name='pool4')(conv4_1)

    up3_2 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up32', padding='same')(conv4_1)
    conv3_2 = concatenate([up3_2, conv3_1], name='merge32', axis=bn_axis)
    conv3_2 = standard_unit(conv3_2, stage='32', nb_filter=nb_filter[2])

    up2_3 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up23', padding='same')(conv3_2)
    conv2_3 = concatenate([up2_3, conv2_1, conv2_2], name='merge23', axis=bn_axis)
    conv2_3 = standard_unit(conv2_3, stage='23', nb_filter=nb_filter[1])

    up1_4 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up14', padding='same')(conv2_3)
    conv1_4 = concatenate([up1_4, conv1_1, conv1_2, conv1_3], name='merge14', axis=bn_axis)
    conv1_4 = standard_unit(conv1_4, stage='14', nb_filter=nb_filter[0])

    conv5_1 = standard_unit(pool4, stage='51', nb_filter=nb_filter[4])

    up4_2 = Conv2DTranspose(nb_filter[3], (2, 2), strides=(2, 2), name='up42', padding='same')(conv5_1)
    conv4_2 = concatenate([up4_2, conv4_1], name='merge42', axis=bn_axis)
    conv4_2 = standard_unit(conv4_2, stage='42', nb_filter=nb_filter[3])

    up3_3 = Conv2DTranspose(nb_filter[2], (2, 2), strides=(2, 2), name='up33', padding='same')(conv4_2)
    conv3_3 = concatenate([up3_3, conv3_1, conv3_2], name='merge33', axis=bn_axis)
    conv3_3 = standard_unit(conv3_3, stage='33', nb_filter=nb_filter[2])

    up2_4 = Conv2DTranspose(nb_filter[1], (2, 2), strides=(2, 2), name='up24', padding='same')(conv3_3)
    conv2_4 = concatenate([up2_4, conv2_1, conv2_2, conv2_3], name='merge24', axis=bn_axis)
    conv2_4 = standard_unit(conv2_4, stage='24', nb_filter=nb_filter[1])

    up1_5 = Conv2DTranspose(nb_filter[0], (2, 2), strides=(2, 2), name='up15', padding='same')(conv2_4)
    conv1_5 = concatenate([up1_5, conv1_1, conv1_2, conv1_3, conv1_4], name='merge15', axis=bn_axis)
    conv1_5 = standard_unit(conv1_5, stage='15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(num_class, (1, 1), name='output_1', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_2)
    nestnet_output_2 = Conv2D(num_class, (1, 1), name='output_2', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_3)
    nestnet_output_3 = Conv2D(num_class, (1, 1), name='output_3', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_4)
    nestnet_output_4 = Conv2D(num_class, (1, 1), name='output_4', kernel_initializer='he_normal',
                              padding='same', kernel_regularizer=l2(1e-4))(conv1_5)

    if deep_supervision:
        output = [nestnet_output_1,nestnet_output_2,nestnet_output_3,nestnet_output_4]
        # model = Model(input=img_input, output=[nestnet_output_1,
        #                                        nestnet_output_2,
        #                                        nestnet_output_3,
        #                                        nestnet_output_4])
    else:
        #model = Model(input=img_input, output=[nestnet_output_4])
        output = [nestnet_output_4]

    return output



def create_conv_net(x, keep_prob, channels, n_class, **kwargs):
    deep_supervision = kwargs.pop("deep_supervision",False)
    return Nest_Net(x,keep_prob,n_class,deep_supervision),tf.global_variables()


class SegModel(object):
    """
    A unet implementation

    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param cost: (optional) name of the cost function. Default is 'cross_entropy'
    :param cost_kwargs: (optional) kwargs passed to the cost function. See Unet._get_cost for more options
    """

    def __init__(self,input_tensor ,mask_tensor ,sess,model_dir,channels=1,n_class=3 ,cost_kwargs={}, **kwargs):
        #tf.reset_default_graph()

        self.n_class = n_class
        self.summaries = kwargs.get("summaries", True)
        self.sess = sess

        self.x = input_tensor
        self.mask = mask_tensor
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_probability")  # dropout (keep probability)
        start_variables = tf.global_variables()
        logits , self.variables = create_conv_net(self.x, self.keep_prob, channels, n_class,**kwargs)
        self.variables = [var for var in self.variables if not var in start_variables]
        self.saver = tf.train.Saver(var_list=self.variables)
        with tf.name_scope("results"):
            self.predicter = pixel_wise_softmax(logits,self.mask)
        ckpt = tf.train.get_checkpoint_state(model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(sess, os.path.join(model_dir, ckpt_name))
    def predict3D(self, sess, x_tests,masks,idex = 0,dstep=1):
        predicts = np.zeros(list(x_tests.shape[:-1])+[self.n_class])
        d = x_tests.shape[0]
        x_test = np.zeros([dstep]+list(x_tests.shape[1:]))
        mask_test = np.zeros([dstep]+list(masks.shape[1:]))
        for i in range(0,d,dstep):
            if(i+dstep<d):
                x_test[0:dstep] = x_tests[i:i+dstep]
                mask_test[0:dstep] = masks[i:i+dstep]
                prediction = sess.run(self.predicter[idex], feed_dict={self.x: x_test, self.mask:mask_test, self.keep_prob: 1.})
                predicts[i:i+dstep] = prediction[0:dstep]
            else:
                x_test[0:d-i] = x_tests[i:d]
                x_test[d-i:dstep] = 0
                mask_test[0:d-i] = masks[0:d-i]
                mask_test[d - i:dstep] = 0
                prediction = sess.run(self.predicter[idex], feed_dict={self.x: x_test,self.mask:mask_test, self.keep_prob: 1.})
                predicts[i:d] = prediction[0:d-i]
        return predicts
class CurUnetppModel(SegModel):
    def __init__(self, sess, model_dir, channels=1, n_class=3, cost_kwargs={},
                 **kwargs):
        self.input_slice = tf.placeholder(tf.float32, [1, 512, 512, 1])
        self.repredict_slice = tf.placeholder(tf.float32, [1, 512, 512, 2])
        input_tensor = self.input_slice * self.repredict_slice[..., 1:]
        self.mask_slice = tf.placeholder(tf.float32, [1, 512, 512, 1])
        mask_tensor = self.mask_slice
        super(CurUnetppModel,self).__init__(input_tensor=input_tensor ,mask_tensor=mask_tensor ,sess=sess,model_dir=model_dir,channels=channels,n_class=n_class ,cost_kwargs=cost_kwargs, **kwargs)
    def __call__(self, input_data,input_repredict,input_mask,idex):
        volume = input_data * input_repredict[..., 1:]
        mask = input_mask
        predict = self.predict3D(self.sess,volume,mask,idex)
        return predict

class CurUnetppModel2(SegModel):
    def __init__(self, sess, model_dir, channels=1, n_class=3, cost_kwargs={},
                 **kwargs):
        self.input_slice = tf.placeholder(tf.float32, [1, 512, 512, 1])
        self.repredict_slice = tf.placeholder(tf.float32, [1, 512, 512, 2])
        #input_tensor = self.input_slice * self.repredict_slice[..., 1:]
        self.mask_slice = tf.placeholder(tf.float32, [1, 512, 512, 1])

        input_tensor0 = self.input_slice
        input_tensor1 = self.input_slice * self.repredict_slice[..., 1:]
        input_tensor2 = self.input_slice * self.repredict_slice[..., 0:1]
        input_tensor = tf.concat([input_tensor0, input_tensor1, input_tensor2], axis=-1)


        input_tensor = input_tensor * self.mask_slice
        mask_tensor = self.mask_slice
        super(CurUnetppModel2,self).__init__(input_tensor=input_tensor ,mask_tensor=mask_tensor ,sess=sess,model_dir=model_dir,channels=channels,n_class=n_class ,cost_kwargs=cost_kwargs, **kwargs)
    def __call__(self, input_data,input_repredict,input_mask,idex):
        volume0 = input_data
        volume1 = input_data * input_repredict[..., 1:]
        volume2 = input_data * input_repredict[..., 0:1]
        volume = np.concatenate([volume0, volume1, volume2], axis=-1)

        mask = input_mask
        volume = volume*mask
        predict = self.predict3D(self.sess,volume,mask,idex)
        return predict
    def predict3D(self, sess, x_tests,masks,idex = 0,dstep=1):
        predicts = np.zeros(list(x_tests.shape[:-1]) + [self.n_class])
        d = x_tests.shape[0]
        x_test = np.zeros([dstep] + list(x_tests.shape[1:]))
        mask_test = np.zeros([dstep] + list(masks.shape[1:]))

        for i in range(0, d, dstep):
            if (i + dstep < d):
                x_test[0:dstep] = x_tests[i:i + dstep]
                mask_test[0:dstep] = masks[i:i + dstep]
                prediction = sess.run(self.predicter[idex],
                                      feed_dict={self.x: x_test, self.mask: mask_test,
                                                 self.keep_prob: 1.})
                predicts[i:i + dstep] = prediction[0:dstep]
            else:
                x_test[0:d - i] = x_tests[i:d]
                x_test[d - i:dstep] = 0
                mask_test[0:d - i] = masks[0:d - i]
                mask_test[d - i:dstep] = 0
                prediction = sess.run(self.predicter[idex],
                                      feed_dict={self.x: x_test,  self.mask: mask_test,
                                                 self.keep_prob: 1.})
                predicts[i:d] = prediction[0:d - i]
        return predicts