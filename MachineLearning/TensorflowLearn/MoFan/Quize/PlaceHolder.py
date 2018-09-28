
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output1 = tf.multiply(input1, input2)

sess = tf.Session()
print(sess.run(output1, feed_dict={input1 : [7.], input2 : [2.]}))







