

import PIL
from PIL import Image
import numpy as np
import os
import tensorflow as tf


def prediction(filename):
    imageSize = 64
    data = []
    print('Resizing')
    img=Image.open("./" + filename)
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img = np.array(img)


    img1=Image.open("./1.jpg")
    img1 = img1.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img1 = np.array(img1)

    data.append(img)
    data.append(img1)
    data = np.array(data)

    print(data.shape)




    x = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])

		

    # 입력 이미지

    X_img = x
    # dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
	
    W1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=0.01))

    b1 = tf.Variable(tf.random_normal([64]))

    L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')+b1
    L1 = tf.nn.relu(L1)
    L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')



    W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev = 0.01))
    b2 = tf.Variable(tf.random_normal([64]))

    L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')+b2

    L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


    L2 = tf.reshape(L2, [-1, 16*16*64])

    W_fc1 = tf.Variable(tf.random_normal([16*16*64, 384], stddev=5e-2))
    b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

    h_fc1 = tf.nn.relu(tf.matmul(L2, W_fc1) + b_fc1)


    W3 = tf.get_variable("W2", shape=[384, 2], initializer=tf.contrib.layers.xavier_initializer())
    b = tf.Variable(tf.random_normal([2]))
    output = tf.matmul(h_fc1, W3) + b







    y = tf.nn.softmax(output)

    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()

    result = 0
    with tf.Session() as sess:
        sess.run(init_op)
        save_path = "./animal2.ckpt"
        saver.restore(sess, save_path)
        predictions = sess.run(y, feed_dict={x: data})
        result = (predictions[0][0])

    tf.reset_default_graph()
    return result
