import PIL
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

# data directory, 경로탐색
input = os.getcwd() + "/data"
test = os.getcwd() + "/test"
vali = os.getcwd() + "/vali"
imageSize = 128
imageDepth = 3
debugEncodedImage = False

xtrain = []
ytrain = []
xtest = []
ytest = []
xval = []
yval = []

# convert to binary bitmap given image and write to law output file
def writeBinaray(imagePath):
    img = Image.open(imagePath)
    img = img.resize((imageSize, imageSize), PIL.Image.ANTIALIAS)
    img = (np.array(img))

    return img

# input directory 안의 파일과 directory를 리스트로 반환

subDirs = os.listdir(input)
valiDirs = os.listdir(vali)
testDirs = os.listdir(test)
numberOfClasses = len(input)


# MAKE train DATA
label = -1

for subDir in subDirs:

    # input data 의 경로 만들기
    subDirPath = os.path.join(input, subDir)

    # filter not directory
    if not os.path.isdir(subDirPath):
        continue

    imageFileList = os.listdir(subDirPath)
    label += 1

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))

    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        img = writeBinaray(imagePath)
        xtrain.append(img)
        ytrain.append(label)

for i in reversed(range(len(xtrain))):

	a = xtrain[i]
	if a.shape != (imageSize,imageSize,3):
		del xtrain[i]
		del ytrain[i]

x_train = np.array(xtrain)
b = np.array(ytrain)
y_train = np.reshape(b, (len(b),1))

# MAKE vali DATA

label = -1

for subDir in valiDirs:

    # input data 의 경로 만들기
    subDirPath = os.path.join(vali, subDir)

    # filter not directory
    if not os.path.isdir(subDirPath):
        continue

    imageFileList = os.listdir(subDirPath)
    label += 1

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))

    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        img = writeBinaray(imagePath)
        xval.append(img)
        yval.append(label)


for i in reversed(range(len(xval))):

	a = xval[i]
	if a.shape != (imageSize,imageSize,3):
		del xval[i], yval[i]
   	


x_vali = np.array(xval)
b = np.array(yval)
y_vali = np.reshape(b, (len(b),1))



# MAKE test DATA

label = -1

for subDir in testDirs:

    # input data 의 경로 만들기
    subDirPath = os.path.join(test, subDir)

    # filter not directory
    if not os.path.isdir(subDirPath):
        continue

    imageFileList = os.listdir(subDirPath)
    label += 1

    print("writing %3d images, %s" % (len(imageFileList), subDirPath))

    for imageFile in imageFileList:
        imagePath = os.path.join(subDirPath, imageFile)
        img = writeBinaray(imagePath)
        xtest.append(img)
        ytest.append(label)


for i in reversed(range(len(xtest))):

	a = xtest[i]
	if a.shape != (imageSize,imageSize,3):
		del xtest[i], ytest[i]
   	
x_test = np.array(xtest)
b = np.array(ytest)
y_test = np.reshape(b, (len(b),1))


def build_CNN(x):

	X_img = x
	# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
	keep_prob = tf.placeholder(tf.float32)
		
	W1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=0.01))
	
	b1 = tf.Variable(tf.random_normal([64]))

	L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')+b1
	L1 = tf.nn.relu(L1)
	L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	L1 = tf.nn.dropout(L1, keep_prob)

	

	W2 = tf.Variable(tf.random_normal([3, 3, 64, 64], stddev = 0.01))
	b2 = tf.Variable(tf.random_normal([64]))

	L2 = tf.nn.conv2d(L1, W2, strides=[1,1,1,1], padding='SAME')+b2
	
	L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

	L2 = tf.nn.dropout(L2, keep_prob)

	L2 = tf.reshape(L2, [-1, 32*32*64])

	W_fc1 = tf.Variable(tf.random_normal([32*32*64, 384], stddev=5e-2))
	b_fc1 = tf.Variable(tf.constant(0.1, shape=[384]))

	h_fc1 = tf.nn.relu(tf.matmul(L2, W_fc1) + b_fc1)

	

	W3 = tf.get_variable("W2", shape=[384, 2], initializer=tf.contrib.layers.xavier_initializer())
	b = tf.Variable(tf.random_normal([2]))
	hypothesis = tf.matmul(h_fc1, W3) + b
	hypothesis = tf.nn.dropout(hypothesis, keep_prob)

	return hypothesis, keep_prob






def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환한다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 2),axis=1)
y_vali_one_hot = tf.squeeze(tf.one_hot(y_vali, 2),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 2),axis=1)

# Input과 Ouput의 차원을 가이드한다.
x = tf.placeholder(tf.float32, [None, imageSize, imageSize, 3])
y_ = tf.placeholder(tf.float32, [None, 2])

# Convolutional Neural Networks(CNN) 그래프를 생성한다.
y_conv, keep_prob = build_CNN(x)


# Cross Entropy를 비용함수(loss function)으로 정의하고, RMSPropOptimizer 이용해서 비용 함수를 최소화한다.
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

# 정확도를 측정한다.
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 하이퍼 파라미터를 정의한다.
max_steps = 10000 # 최대 몇 step을 학습할지를 정한다. 


with tf.Session() as sess:
  # 모든 변수들을 초기화한다. 
  sess.run(tf.global_variables_initializer())
  
  # 20000번 학습(training)을 진행한다.
  for i in range(max_steps):
    batch = next_batch(100, x_train, y_train_one_hot.eval())

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력한다.
    if i % 100 == 0:

      batch_vali = next_batch(200, x_vali, y_vali_one_hot.eval())

      validation_accuracy = accuracy.eval(feed_dict={ x: batch_vali[0], y_: batch_vali[1], keep_prob: 1.0})
      loss = cross_entropy.eval(feed_dict={ x: batch_vali[0], y_: batch_vali[1], keep_prob: 1.0})

      print('step %d, validation accuracy %g, loss %g' % (i, validation_accuracy, loss))
    # 20% 확률의 Dropout을 이용해서 학습을 진행한다.
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.9})

  saver = tf.train.Saver()
  saver.save(sess, "./animal.ckpt")

  test_batch = next_batch(200, x_test, y_test_one_hot.eval())
  # 테스트 데이터에 대한 정확도를 출력한다.
  print('test accuracy %g' % accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))

