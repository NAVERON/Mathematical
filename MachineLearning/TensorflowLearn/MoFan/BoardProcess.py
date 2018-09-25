
import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

#模拟原始数据
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

def add_layer(inputs, in_size, out_size, n_layer, activation_function = None):
    layer_name = "layer%s"%n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name = "W")
            tf.summary.histogram(layer_name + "/weights", Weights)
        with tf.name_scope("biases"):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name = "b")
            tf.summary.histogram(layer_name + "/biases", biases)
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
        
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        
        tf.summary.histogram(layer_name + "/outputs", outputs)
    
    return outputs


xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 构建神经网络模型
layer_one = add_layer(xs, 1, 10, n_layer=1, activation_function = tf.nn.relu)
prediction = add_layer(layer_one, 10, 1, n_layer=2, activation_function = None)

#设置损失率
with tf.name_scope("loss"):
    loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.square(ys - prediction), reduction_indices=[1]
            )
        )
    tf.summary.scalar("loss", loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#综合处理数据
sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(tf.global_variables_initializer())


for i in range(1000):
    sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
    if i % 50 == 0:
        rs = sess.run(merged, feed_dict={xs:x_data, ys:y_data})
        writer.add_summary(rs, i)
















