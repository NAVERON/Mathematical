


import tensorflow as tf
import numpy as np
import os



os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
#实际数据，即标记的数据，使用数据来训练模型
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1+0.3

#模型
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))
y = Weights * x_data + biases

loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))




