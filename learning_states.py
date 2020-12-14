import tensorflow as tf
import pandas as pd
import numpy as np


def split_train_test(data, div=0.8):
    xdim = data.shape[0]
    split = int(xdim * div)
    train = data[:split]
    test = data[split:]
    return train, test



col_list = ["States", "Front", "Right", "Tof"]  # define columns to read  from csv TODO add left
df = pd.read_csv("drone.csv", usecols=col_list)  # read the collums from the defind col_list
one_hot_state = pd.get_dummies(df["States"])  # transform the data to be predicted to one hot
# one_hot = tf.one_hot(category_indices, unique_category_count) #tensorflow way

data_list = ["Front", "Right", "Tof"]  # define columns to read  from csv TODO add left
df_data = pd.read_csv("drone.csv", usecols=data_list)  # read the collums from the defind col_list
data = df_data.to_numpy()
#data = np.random.shuffle(data) #TODO shuffle the arr
state = one_hot_state.to_numpy()  # make a numpy array of states
#state = np.random.shuffle(state)

(train_data, test_data) = split_train_test(data)
(train_states,test_states) = split_train_test(state)

# front_arr = np.array([front]) if i wont ro predict by one line only need to convert it to a 2d array

left = df["Right"].to_numpy()
button = df["Tof"].to_numpy()

labels_amount = 4  # the amount of labels , to be predicted
features_amount = 3  # The amount of data that the prediction is going to be made on (raw data)
learing_rate = 0.0005

x = tf.placeholder(tf.float32, (None, features_amount))
y_ = tf.placeholder(tf.float32, (None, labels_amount))
W = tf.Variable(tf.zeros((features_amount, labels_amount)))
b = tf.Variable(tf.zeros((labels_amount)))  # TODO change it to tf.random.uniform
y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(data)
for i in range(100000):
    sess.run(train_step, feed_dict={x: train_data, y_: train_states})  # BGD

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_data, y_: test_states}))

#Cannot feed value of shape (2175,) for Tensor 'Placeholder:0', which has shape '(?, 3)'