
# coding: utf-8

# In[1]:


import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from captcha.image import ImageCaptcha
from datetime import datetime


# In[2]:


# initialize the arrays and constants

NUMBER = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
LOWER_CASE = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
              'n', 'o', 'p', 'q', 'r', 's', 't', 'u','v', 'w', 'x', 'y', 'z']
CAPTCHA_LIST = NUMBER + LOWER_CASE
CAPTCHA_LEN = 4
CAPTCHA_HEIGHT = 60
CAPTCHA_WIDTH = 160


# In[3]:


# randomly choose four elements from captcha list which contains number and lower case characters

def random_captcha_text():
    captcha_text = [random.choice(CAPTCHA_LIST) for i in range(CAPTCHA_LEN)]
    return ''.join(captcha_text)


# In[4]:


# use captcha to generate the text and image which is a numpy array

def gen_captcha_text_and_image():
    image = ImageCaptcha(width=CAPTCHA_WIDTH, height=CAPTCHA_HEIGHT)
    
    captcha_text = random_captcha_text()
    captcha = image.generate(captcha_text)
    
    captcha_image = Image.open(captcha)
    
    captcha_image = np.array(captcha_image)
    return captcha_text, captcha_image


# In[5]:


# testing generated text and image

if __name__ == '__main__':
    text, image = gen_captcha_text_and_image()
 
    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)
 
    plt.show()


# In[6]:


# convert 3D rgb to 1D grayscale for simplification

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


# In[7]:


# convert text to vector for convolution

def text2vec(text): 
    text_len = len(text)
    vector = np.zeros(CAPTCHA_LEN*len(CAPTCHA_LIST))
    for i in range(text_len): 
        vector[CAPTCHA_LIST.index(text[i])+i*len(CAPTCHA_LIST)] = 1
    return vector


# In[8]:


def next_batch(batch_size=100):
    batch_x = np.zeros([batch_size, CAPTCHA_HEIGHT * CAPTCHA_WIDTH])
    batch_y = np.zeros([batch_size, CAPTCHA_LEN * len(CAPTCHA_LIST)])
 
    for i in range(batch_size):
        text, image = gen_captcha_text_and_image()
        image = rgb2gray(image)
#         plt.imshow(image, cmap = plt.get_cmap('gray'))
#         plt.show()
        batch_x[i,:] = image.flatten() / 255 # standardize to 0-1 range since color uses 0-255 values
        batch_y[i,:] = text2vec(text)

    return batch_x, batch_y


# In[9]:


if __name__ == '__main__':
    x, y = next_batch(batch_size=1)
    print(x)
    print(y)


# In[10]:


import tensorflow as tf


# In[11]:


X = tf.placeholder(tf.float32, [None, 60*160])
Y = tf.placeholder(tf.float32, [None, 4*4])
keep_prob = tf.placeholder(tf.float32)


# In[12]:


def weight(shape):
    initial = 0.01 * tf.random_normal(shape)
    return tf.Variable(initial)


# In[13]:


def bias(shape):
    initial = 0.01 * tf.random_normal(shape)
    return tf.Variable(initial)


# In[14]:


def conv(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


# In[15]:


def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


# In[17]:


def cnn_model(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    w_conv1 = weight([3, 3, 1, 32])
    b_conv1 = bias([32])
    # ReLU
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv(x_image, w_conv1), b_conv1))
    # pooling
    h_pool1 = max_pool(h_conv1)
    # dropout
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight([3, 3, 32, 64])
    b_conv2 = bias([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # layer 3
    w_conv3 = weight([3, 3, 64, 64])
    b_conv3 = bias([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)

    # full connected layer
    image_height = int(h_drop3.shape[1])
    image_width = int(h_drop3.shape[2])
    w_fc = weight([image_height*image_width*64, 1024])
    b_fc = bias([1024])
    h_drop3_re = tf.reshape(h_drop3, [-1, image_height*image_width*64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop3_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # output layer
    w_out = weight([1024, len(captcha_list)*captcha_len])
    b_out = bias([len(captcha_list)*captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv


# In[18]:


def cnn_graph2(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    # rulu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # layer 3
    w_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])
    h_conv3 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop2, w_conv3), b_conv3))
    h_pool3 = max_pool_2x2(h_conv3)
    h_drop3 = tf.nn.dropout(h_pool3, keep_prob)
    
    # layer 4
    w_conv4 = weight_variable([3, 3, 64, 64])
    b_conv4 = bias_variable([64])
    h_conv4 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop3, w_conv4), b_conv4))
    h_pool4 = max_pool_2x2(h_conv4)
    h_drop4 = tf.nn.dropout(h_pool4, keep_prob)

    # full connect layer
    image_height = int(h_drop4.shape[1])
    image_width = int(h_drop4.shape[2])
    w_fc = weight_variable([image_height*image_width*64, 1024])
    b_fc = bias_variable([1024])
    h_drop4_re = tf.reshape(h_drop4, [-1, image_height*image_width*64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop4_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # out layer
    w_out = weight_variable([1024, len(captcha_list)*captcha_len])
    b_out = bias_variable([len(captcha_list)*captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv


# In[16]:


def cnn_graph3(x, keep_prob, size, captcha_list=CAPTCHA_LIST, captcha_len=CAPTCHA_LEN):
    # 图片reshape为4维向量
    image_height, image_width = size
    x_image = tf.reshape(x, shape=[-1, image_height, image_width, 1])

    # layer 1
    # filter定义为3x3x1， 输出32个特征, 即32个filter
    w_conv1 = weight_variable([3, 3, 1, 32])
    b_conv1 = bias_variable([32])
    # rulu激活函数
    h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, w_conv1), b_conv1))
    # 池化
    h_pool1 = max_pool_2x2(h_conv1)
    # dropout防止过拟合
    h_drop1 = tf.nn.dropout(h_pool1, keep_prob)

    # layer 2
    w_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2d(h_drop1, w_conv2), b_conv2))
    h_pool2 = max_pool_2x2(h_conv2)
    h_drop2 = tf.nn.dropout(h_pool2, keep_prob)

    # full connect layer
    image_height = int(h_drop2.shape[1])
    image_width = int(h_drop2.shape[2])
    w_fc = weight_variable([image_height*image_width*64, 1024])
    b_fc = bias_variable([1024])
    h_drop2_re = tf.reshape(h_drop2, [-1, image_height*image_width*64])
    h_fc = tf.nn.relu(tf.add(tf.matmul(h_drop2_re, w_fc), b_fc))
    h_drop_fc = tf.nn.dropout(h_fc, keep_prob)

    # out layer
    w_out = weight_variable([1024, len(captcha_list)*captcha_len])
    b_out = bias_variable([len(captcha_list)*captcha_len])
    y_conv = tf.add(tf.matmul(h_drop_fc, w_out), b_out)
    return y_conv


# In[17]:


def optimizer(y, y_conv):
    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_conv, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
    return optimizer


# In[18]:


def accuracy(y, y_conv, width=len(CAPTCHA_LIST), height=CAPTCHA_LEN):
    predict = tf.reshape(y_conv, [-1, height, width])
    max_predict_idx = tf.argmax(predict, 2)
    label = tf.reshape(y, [-1, height, width])
    max_label_idx = tf.argmax(label, 2)
    correct_p = tf.equal(max_predict_idx, max_label_idx)
    accuracy = tf.reduce_mean(tf.cast(correct_p, tf.float32))
    return accuracy


# In[21]:


def train(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.95
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph(x, keep_prob, (height, width))
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    while 1:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                break
        step += 1
    sess.close()


# In[22]:


def train2(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.95
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph2(x, keep_prob, (height, width))
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    while 1:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                break
        step += 1
    sess.close()


# In[19]:


def train3(height=CAPTCHA_HEIGHT, width=CAPTCHA_WIDTH, y_size=len(CAPTCHA_LIST)*CAPTCHA_LEN):
    # cnn在图像大小是2的倍数时性能最高, 如果图像大小不是2的倍数，可以在图像边缘补无用像素
    # 在图像上补2行，下补3行，左补2行，右补2行
    # np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))

    acc_rate = 0.95
    # 按照图片大小申请占位符
    x = tf.placeholder(tf.float32, [None, height * width])
    y = tf.placeholder(tf.float32, [None, y_size])
    # 防止过拟合 训练时启用 测试时不启用
    keep_prob = tf.placeholder(tf.float32)
    # cnn模型
    y_conv = cnn_graph3(x, keep_prob, (height, width))
    # 最优化
    optimizer = optimize_graph(y, y_conv)
    # 偏差
    accuracy = accuracy_graph(y, y_conv)
    # 启动会话.开始训练
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    step = 0
    while 1:
        batch_x, batch_y = next_batch(64)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.75})
        # 每训练一百次测试一次
        if step % 100 == 0:
            batch_x_test, batch_y_test = next_batch(100)
            acc = sess.run(accuracy, feed_dict={x: batch_x_test, y: batch_y_test, keep_prob: 1.0})
            print(datetime.now().strftime('%c'), ' step:', step, ' accuracy:', acc)
            # 偏差满足要求，保存模型
            if acc > acc_rate:
                break
        step += 1
    sess.close()


# In[56]:


if __name__ == '__main__':
    train()


# In[23]:


if __name__ == '__main__':
    train2()


# In[20]:


if __name__ == '__main__':
    train3()

