'''
100 次 validation acc: 0.875000
'''

import os
import glob
import time
import numpy as np
import tensorflow as tf
from skimage import io, transform

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '1'
# 这是默认的显示等级，显示所有信息
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# 只显示 warning 和 Error
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
# 只显示 Error


# 读取图片
def read_img(path, w, h):
    cate = [path + x for x in os.listdir(path) if os.path.isdir(path + x)]
    # print(cate)

    imgs = []
    labels = []

    print('Start read the image ...')

    for index, folder in enumerate(cate):
        # print(index, folder)
        for im in glob.glob(folder + '/*.jpg'):
            # print('Reading The Image: %s' % im)
            img = io.imread(im)
            # img = transform.resize(img, (w, h))
            imgs.append(img)
            # print(index)
            labels.append(index)                    # 每个子文件夹是一个label
            # print(len(labels))

    print('Finished ...')
    print(len(imgs))
    print(len(labels))

    return np.asarray(imgs, np.float32), np.asarray(labels, np.float32)


# 打乱顺序 为了均匀取数据
def messUpOrder(data, label):
    num_example = data.shape[0]
    arr = np.arange(num_example)
    np.random.shuffle(arr)
    data = data[arr]
    label = label[arr]

    return data, label


# 将所有数据分为训练集和验证集
def segmentation(data, label, ratio=0.8):
    num_example = data.shape[0]
    s = np.int(num_example * ratio)
    x_train = data[:s]
    y_train = label[:s]
    x_val = data[s:]
    y_val = label[s:]

    return x_train, y_train, x_val, y_val


# 构建网络
def buildCNN(w, h, c):
    # 占位符
    x = tf.placeholder(tf.float32, shape=[None, w, h, c], name='x')
    y_ = tf.placeholder(tf.int32, shape=[None, ], name='y_')

    # 第一个卷积层 + 池化层（100——>50)
    conv1 = tf.layers.conv2d(
        inputs=x,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # 第二个卷积层 + 池化层 (50->25)
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # 第三个卷积层 + 池化层 (25->12)
    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    # 第四个卷积层 + 池化层 (12->6)
    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=128,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu,
        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01))
    pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=[2, 2], strides=2)

    re1 = tf.reshape(pool4, [-1, 6 * 6 * 128])

    # 全连接层
    dense1 = tf.layers.dense(inputs=re1,
                             units=1024,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    dense2 = tf.layers.dense(inputs=dense1,
                             units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))
    logits = tf.layers.dense(inputs=dense2,
                             units=32,
                             activation=None,
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                             kernel_regularizer=tf.contrib.layers.l2_regularizer(0.003))

    return logits, x, y_


# 返回损失函数的值，准确值等参数
def accCNN(logits, y_):
    loss = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=logits)
    train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), y_)
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    return loss, train_op, correct_prediction, acc


# 定义一个函数，按批次取数据
def minibatches(inputs=None, targets=None, batch_size=None, shuffle=False):
    assert len(inputs) == len(targets), 'len(inputs) != len(targets)'
    if shuffle:
        indices = np.arange(len(inputs))   # [0, 1, 2, ..., 422]
        np.random.shuffle(indices)     # 打乱下标顺序
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

# 训练和测试
def runable(x_train, y_train, train_op, loss, acc, x, y_, x_val, y_val):
    # 训练和测试数据，可将n_epoch设置更大一些
    n_epoch = 100
    batch_size = 20
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    for epoch in range(n_epoch):
        print('epoch: ', epoch)
        # training
        train_loss, train_acc, n_batch = 0, 0, 0
        for x_train_a, y_train_a in minibatches(x_train, y_train, batch_size, shuffle=True):
            # print('x_val_a: ', x_val_a.shape())
            # print('y_val_a: ', y_val_a.shape())
            _, err, ac = sess.run([train_op, loss, acc], feed_dict={x: x_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
        print("train loss: %f" % (train_loss / n_batch))
        print("train acc: %f" % (train_acc / n_batch))

        # validation
        val_loss, val_acc, n_batch = 0, 0, 0
        for x_val_a, y_val_a in minibatches(x_val, y_val, batch_size, shuffle=False):
            # print('x_val_a: ', x_val_a)
            # print('y_val_a: ', y_val_a)
            err, ac = sess.run([loss, acc], feed_dict={x: x_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("validation loss: %f" % (val_loss / n_batch))
        print("validation acc: %f" % (val_acc / n_batch))
        print('*' * 50)

    sess.close()


if __name__ == '__main__':
    imgpath = 'dataset/'

    w = 100
    h = 100
    c = 3

    ratio = 0.8  # 选取训练集的比例

    data, label = read_img(path=imgpath, w=w, h=h)
    print(0)

    data, label = messUpOrder(data=data, label=label)
    print(1)

    x_train, y_train, x_val, y_val = segmentation(data=data, label=label, ratio=ratio)
    print('x_train: ', len(x_train))
    print('y_train: ', len(y_train))
    print(2)

    logits, x, y_ = buildCNN(w=w, h=h, c=c)
    print(3)
    print(logits)

    loss, train_op, correct_prediction, acc = accCNN(logits=logits, y_=y_)
    print(4)

    runable(x_train=x_train, y_train=y_train, train_op=train_op, loss=loss, acc=acc, x=x, y_=y_, x_val=x_val, y_val=y_val)