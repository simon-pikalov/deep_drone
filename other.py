import numpy as np
import tensorflow as tf

# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

features = 84
categories = 7

layers_size = [100, 50]


def one_hot_data(data):
    size_x = data.shape[0]
    size_y = data.shape[1] - 2
    d = np.zeros((size_x, 2 * size_y))
    for i, row in enumerate(data):
        for j, value in enumerate(row[:-2]):
            if value == row[-2]:
                d[i][j] = 1
            elif value == row[-1]:
                d[i][size_y + j] = 1
    print("one hot is ",d)
    return d


def split_train_test(data, div=0.8):
    xdim = data.shape[0]
    split = int(xdim * div)
    train = data[:split]
    test = data[split:]
    return train, test


# TensorFlow variables

x = tf.placeholder(tf.float32, [None, features])
y_ = tf.placeholder(tf.float32, [None, categories])

# Hidden layers
W = []
b = []
z = [x]

last_size = features
for i, size in enumerate(layers_size):
    W += [tf.Variable(tf.truncated_normal([last_size, size], stddev=0.1))]
    b += [tf.Variable(tf.constant(0.1, shape=[size]))]
    z += [tf.nn.relu(tf.matmul(z[i], W[i]) + b[i])]
    last_size = size

# Regular variables

Wn = tf.Variable(tf.truncated_normal([last_size, categories], stddev=0.1))
bn = tf.Variable(tf.constant(0.1, shape=[categories]))

y = tf.nn.softmax(tf.matmul(z[len(layers_size)], Wn) + bn)

# loss = -tf.reduce_mean(y_ * tf.log(y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=tf.matmul(z[len(layers_size)], Wn) + bn))

update = tf.train.GradientDescentOptimizer(0.02).minimize(loss)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Data

data = np.loadtxt("games.csv", delimiter=",", skiprows=1, dtype=int)
np.random.shuffle(data)

data_x_all = one_hot_data(data[:, :-1])
data_y_all = np.zeros((data.shape[0], categories), dtype=int)

for index, value in enumerate(data[:, -1:]):
    data_y_all[index, value] = 1

data_x_train, data_x_test = split_train_test(data_x_all)
data_y_train, data_y_test = split_train_test(data_y_all)

# Start session

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(accuracy, feed_dict={x: data_x_test, y_: data_y_test}))

for i in range(0, 5000):
    if i % 500 == 0:
        [updt, train_loss, train_accuracy] = sess.run([update, loss, accuracy],
                                                      feed_dict={x: data_x_train, y_: data_y_train})
        test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict={x: data_x_test, y_: data_y_test})
        print("Iteration ", i)
        print("Train loss: ", train_loss, ", Test loss: ", test_loss)
        print("Train accuracy: ", train_accuracy, ", Test accuracy: ", test_accuracy)
    else:
        sess.run(update, feed_dict={x: data_x_train, y_: data_y_train})

# Results:

# [100,50]
# 10,000 iterations:
# Train loss: 0.9315153, Test loss: 1.0385437
# Test Accuracy: 0.6218531
# 100,000 iterations:
# Train loss:  0.16806039 , Test loss:  2.1166186
# Test Accuracy:  0.66156244
