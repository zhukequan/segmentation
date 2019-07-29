import tensorflow as tf
import numpy as np
class Swell(object):
    def __init__(self,sess):
        kernel_data = np.zeros((21, 21, 21), np.float32)
        for i in range(-10, 11):
            for j in range(-10, 11):
                for k in range(-10, 11):
                    if (i ** 2 + j ** 2 + k ** 2) < 100:
                        kernel_data[10 + i, 10 + j, 10 + k] = 1

        self.input_holder = tf.placeholder(tf.float32, [1, None, None, None, 1])
        kernel = tf.constant(kernel_data)
        kernel = kernel[..., tf.newaxis, tf.newaxis]
        swell_output = tf.nn.conv3d(self.input_holder, kernel, [1, 1, 1, 1, 1], padding="SAME")
        swell_output = swell_output > 0
        # swell_output = tf.squeeze(swell_output)
        self.swell_output = tf.cast(swell_output, tf.float32)
        self.sess = sess
    def __call__(self, input_data):
        swell = self.sess.run(self.swell_output,feed_dict={self.input_holder:input_data})
        return swell