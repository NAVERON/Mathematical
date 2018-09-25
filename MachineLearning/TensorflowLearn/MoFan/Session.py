

import tensorflow as tf
import numpy as np
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2], [2]])
product = tf.matmul(matrix1, matrix2)    #  matrix multiply

#激活会话  有两个方法

#方法1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()


#方法2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)





