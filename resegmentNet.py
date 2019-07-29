import tensorflow as tf
import numpy as np
import os,shutil
#from matplotlib import pyplot as plt
import threading
from multiprocessing import cpu_count
import time

class Net():
    def __init__(self,batch, sess, subdivisions, depth, height, width,model_dir, is_training = True):
        self.batch = batch
        self.subdivisions = subdivisions
        self.batch_subdivisions = int(np.ceil(batch / subdivisions))
        self.depth = depth
        self.height = height
        self.width = width

        self.is_training = is_training
        self.model_dir = model_dir



        self.input_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_subdivisions, self.depth, self.height,
                                                       self.width, 1])
        self.label_placeholder = tf.placeholder(dtype=tf.float32,
                                                shape=[self.batch_subdivisions, self.depth, self.height,
                                                       self.width, 1])
        self.output_Slice = self.net_slice(self.input_placeholder)
        self.stored_value = [v for v in tf.global_variables() if 'slice' in v.name]
        #os.environ["CUDA_VISIBLE_DEVICES"]="1"
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
        self.sess = sess
        self.saver = tf.train.Saver(var_list=self.stored_value)
        tf.global_variables_initializer().run()
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(self.model_dir,ckpt_name) )

    def make_var(self, name, shape):
        return tf.get_variable(name=name,shape=shape)
    def bn(self, input, name, epsilon=1e-5):
        # ci = input.get_shape()[-1].value
        # shape = input.get_shape()
        # beta = tf.get_variable(name=name + 'beta', shape=ci, initializer=tf.constant_initializer(0.0), trainable=True)
        # gamma = tf.get_variable(name=name + 'gamma', shape=ci, initializer=tf.constant_initializer(1.0), trainable=True)
        # axises = list(range(len(shape) - 1))
        # batch_mean, batch_var = tf.nn.moments(input, axises, name='moments')
        # return tf.nn.batch_normalization(input, batch_mean, batch_var, offset=beta, scale=gamma,
        #                                 variance_epsilon=epsilon,
        #                                 name="bn")
        return tf.layers.batch_normalization(input, momentum=0.99, epsilon=epsilon, center=True, scale=True,
                                             training=self.is_training, name=name)

    def conv2d(self,input,kh,kw,co,sh,sw,name, biased=False, bn=True, relu=True):
        ci = input.get_shape()[-1].value
        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, sh, sw, 1], padding='SAME')
        kernel = self.make_var(name+'weights', [kh, kw, ci, co])
        output = convolve(input, kernel)
        if biased:
            biases = self.make_var(name+'biases', [co])
            output = tf.nn.bias_add(output, biases)
        if bn:
            output = self.bn(output, name)
        if relu:
            output = tf.nn.relu(output)
        return output
    def max_pool2d(self, input,  kh, kw,  sh, sw, name):
        return tf.nn.max_pool(input,[1,kh,kw,1],[1,sh,sw,1],padding='SAME',name=name)
    def deconv2d(self, input, kh,kw,co,sh,sw,name,batch=1,biased=False,bn=True,relu=True):
        ci = input.get_shape()[-1].value
        height_in = input.get_shape()[1].value
        width_in = input.get_shape()[2].value
        kernel = self.make_var(name+'weights', [kh, kw, co, ci])
        output = tf.nn.conv2d_transpose(input, kernel, [batch, sh * height_in, sw * width_in, co], [1, sh, sw, 1],
                                        'SAME')
        if biased:
            biases = self.make_var(name+'biases', [co])
            output = tf.nn.bias_add(output, biases)
        if bn:
            output = self.bn(output,name)
        if relu:
            output = tf.nn.relu(output)
        return output
    def conv3d(self,input,kd,kh,kw,co,sd,sh,sw,name, biased=False, bn=True, relu=True):
        ci = input.get_shape()[-1].value
        pad_d = int((kd - 1) / 2)
        pad_h = int((kh - 1) / 2)
        pad_w = int((kw - 1) / 2)
        input = tf.pad(input,[[0,0],[pad_d,pad_d],[pad_h,pad_h],[pad_w,pad_w],[0,0]],'SYMMETRIC')
        convolve = lambda i, k: tf.nn.conv3d(i, k, [1, sd, sh, sw, 1], padding='VALID')
        kernel = self.make_var(name + 'weights', [kd, kh, kw, ci, co])
        output = convolve(input, kernel)
        if biased:
            biases = self.make_var(name + 'biases', [co])
            output = tf.nn.bias_add(output, biases)
        if bn:
            output = self.bn(output, name)
        if relu:
            output = tf.nn.relu(output)
        return output
    def max_pool3d(self, input, kd, kh, kw, sd, sh, sw, name):
        return tf.nn.max_pool3d(input,[1,kd,kh,kw,1],[1,sd,sh,sw,1],padding='SAME',name=name)
    def deconv3d(self, input,kd, kh,kw,co,sd,sh,sw,name,batch,biased=False,bn=True,relu=True):
        ci = input.get_shape()[-1].value
        depth_in = input.get_shape()[1].value
        height_in = input.get_shape()[2].value
        width_in = input.get_shape()[3].value
        kernel = self.make_var(name + 'weights', [kd, kh, kw, co, ci])
        output = tf.nn.conv3d_transpose(input, kernel, [batch, sd*depth_in, sh*height_in, sw*width_in, co], [1, sd, sh, sw, 1], 'SAME')
        if biased:
            biases = self.make_var(name + 'biases', [co])
            output = tf.nn.bias_add(output, biases)
        if bn:
            output = self.bn(output, name)
        if relu:
            output = tf.nn.relu(output)
        return output

    def add(self,input1,input2,name ,bn=True,relu=True):
        output = input1 + input2
        if bn:
            output = self.bn(output, name)
        if relu:
            output = tf.nn.relu(output)
        return output

    def net_slice(self, input, name='slice', reuse=False):
        with tf.variable_scope(name,reuse=reuse):
            input = tf.reshape(input, [self.batch_subdivisions * self.depth, int(self.height/2), 2, int(self.width/2), 2])
            input = tf.transpose(input, [0,1,3,2,4])
            input = tf.reshape(input, [self.batch_subdivisions * self.depth, int(self.height/2), int(self.width/2), 4])

            self.slice_conv11 = self.conv2d(input, 3,3,32,1,1, 'conv11')
            self.slice_conv12 = self.conv2d(self.slice_conv11, 3,3,32,1,1, 'conv12',bn=False,relu=False)
            self.slice_add1 = self.add(self.slice_conv11,self.slice_conv12,"add1")
            self.slice_pool1 = self.conv2d(self.slice_add1, 3,3,32,2,2, 'pool1')

            reshape_2dto3d = tf.reshape(self.slice_pool1, [self.batch_subdivisions, self.depth, int(self.height/4), int(self.width/4), 32])

            self.slice_conv21 = self.conv3d(reshape_2dto3d, 3,3,3,64,1,1,1, 'conv21')
            self.slice_conv22 = self.conv3d(self.slice_conv21, 3,3,3,64,1,1,1, 'conv22',bn=False,relu=False)
            self.slice_add2 = self.add(self.slice_conv21, self.slice_conv22, "add2")
            self.slice_pool2 = self.conv3d(self.slice_add2, 3,3,3,128,2,2,2, 'pool2')

            self.slice_conv31 = self.conv3d(self.slice_pool2, 3,3,3,128,1,1,1, 'conv31')
            self.slice_conv32 = self.conv3d(self.slice_conv31, 3,3,3,128,1,1,1, 'conv32',bn=False,relu=False)
            self.slice_add3 = self.add(self.slice_pool2, self.slice_conv32, "add3")
            self.slice_pool3 = self.conv3d(self.slice_add3, 3,3,3,256,2,2,2, 'pool3')

            self.slice_conv41 = self.conv3d(self.slice_pool3, 3,3,3,256,1,1,1, 'conv41')
            self.slice_conv42 = self.conv3d(self.slice_conv41, 3,3,3,256,1,1,1, 'conv42',bn=False,relu=False)
            self.slice_add4 = self.add(self.slice_pool3, self.slice_conv42, "add4")

            self.slice_pool3_back = self.deconv3d(self.slice_add4, 3,3,3,128,2,2,2, 'pool3_back', self.batch_subdivisions)
            self.slice_cat3_back = tf.concat([self.slice_pool3_back, self.slice_add3], 4)
            self.slice_conv31_back = self.conv3d(self.slice_cat3_back, 3,3,3,128,1,1,1, 'conv31_back')
            self.slice_conv32_back = self.conv3d(self.slice_conv31_back, 3,3,3,128,1,1,1, 'conv32_back',bn=False,relu=False)
            self.slice_add3_back = self.add(self.slice_pool3_back,self.slice_conv32_back,"add3_back")

            self.slice_pool2_back = self.deconv3d(self.slice_add3_back, 3,3,3,64,2,2,2, 'pool2_back',self.batch_subdivisions)
            self.slice_cat2_back = tf.concat([self.slice_pool2_back, self.slice_add2], 4)
            self.slice_conv21_back = self.conv3d(self.slice_cat2_back, 3,3,3,64,1,1,1, 'conv21_back')
            self.slice_conv22_back = self.conv3d(self.slice_conv21_back, 3,3,3,64,1,1,1, 'conv22_back',bn=False,relu=False)
            self.slice_add2_back = self.add(self.slice_pool2_back, self.slice_conv22_back, "add2_back")

            reshape_3dto2d = tf.reshape(self.slice_add2_back, [self.batch_subdivisions * self.depth, int(self.height/4), int(self.width/4), 64])

            self.slice_pool1_back = self.deconv2d(reshape_3dto2d, 3,3,32,2,2, 'pool1_back', self.batch_subdivisions * self.depth)
            self.slice_cat1_back = tf.concat([self.slice_pool1_back, self.slice_add1], 3)
            self.slice_conv11_back = self.conv2d(self.slice_cat1_back, 3,3,32,1,1, 'conv11_back')
            self.slice_conv12_back = self.conv2d(self.slice_conv11_back, 3,3,32,1,1, 'conv12_back',bn=False,relu=False)
            self.slice_add1_back = self.add(self.slice_pool1_back, self.slice_conv12_back, "add1_back")

            self.slice_pool0_back = self.deconv2d(self.slice_add1_back, 3,3,2,2,2, 'pool0_back', self.batch_subdivisions * self.depth, relu=False, bn=False, biased=True)
            self.slice_sigm = tf.nn.softmax(self.slice_pool0_back, name='sigm')
            self.slice_out = tf.clip_by_value(self.slice_sigm, 1e-6, 0.99999, 'out')

            output = tf.reshape(self.slice_out, [self.batch_subdivisions, self.depth, self.height, self.width, 2])
        return output

    def test_slice(self, x_arr,step):
        d0, h0, w0 = x_arr.shape
        loop = (d0)//step
        xarr = np.pad(x_arr,[[0,loop*step+self.depth-d0],[0,0],[0,0]],'constant')
        predict = np.expand_dims(np.zeros_like(xarr),axis=-1)
        predict = np.concatenate([predict,predict],axis=-1)
        for i in range(0,loop,self.batch_subdivisions):
            input_data = np.zeros((self.batch_subdivisions,self.depth,self.height,self.width))
            for j in range(self.batch_subdivisions):
                input_data[j] = xarr[(i+j)*step:(i+j)*step+self.depth]
            input_data = np.expand_dims(input_data,axis=-1)
            out_data = self.sess.run(self.output_Slice,feed_dict={self.input_placeholder:input_data})
            for j in range(self.batch_subdivisions):
                predict[(i+j)*step:(i+j)*step+self.depth] = predict[(i+j)*step:(i+j)*step+self.depth]+out_data[j]
        input_data = np.zeros((self.batch_subdivisions, self.depth, self.height, self.width))
        input_data = np.expand_dims(input_data, axis=-1)
        for j in range(i+1,loop):
            input_data[j-i-1] = xarr[j * step: j * step + self.depth]
        out_data = self.sess.run(self.output_Slice, feed_dict={self.input_placeholder: input_data})

        for j in range(i+1,loop):
            predict[j * step:j * step + self.depth] = predict[j * step:j * step + self.depth] + \
                                                                  out_data[j-i-1]
        predict = predict[0:d0]
        predict = predict/np.sum(predict,axis=-1,keepdims=True)
        return predict