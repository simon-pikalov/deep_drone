import tensorflow as tf
import pandas as pd
import numpy as np


def split_train_test(data, div=0.8):
    xdim = data.shape[0]
    split = int(xdim * div)
    train = data[:split]
    test = data[split:]
    return train, test

# good values itr = 100000 ,learing_rate = 0.0009

hidden1_size, hidden2_size = 100, 50

file_name = "comb.csv"

col_list = ["State"]  # define columns to read  from csv TODO add left
df = pd.read_csv(file_name, usecols=col_list)  # read the collums from the defind col_list
one_hot_state = pd.get_dummies(df["State"])  # transform the data to be predicted to one hot
# one_hot = tf.one_hot(category_indices, unique_category_count) #tensorflow way

data_list = ["VX", "VY", "Range_Front","Range_Right","Range_Left"]  # define columns to read  from csv TODO add left
df_data = pd.read_csv(file_name, usecols=data_list)  # read the collums from the defind col_list
data = df_data.to_numpy()
state = one_hot_state.to_numpy()  # make a numpy array of states

# np.random.shuffle(data)
# np.random.shuffle(state)

print(data)

#state = np.random.shuffle(state)

(train_data, test_data) = split_train_test(data)
(train_states,test_states) = split_train_test(state)

# front_arr = np.array([front]) if i wont ro predict by one line only need to convert it to a 2d array

labels_amount =  state.shape[1] # the amount of labels , to be predicted
features_amount = len(data_list) # The amount of data that the prediction is going to be made on (raw data)
learing_rate = 0.0009


x = tf.placeholder(tf.float32, (None, features_amount))
y_ = tf.placeholder(tf.float32, (None, labels_amount))
W1 = tf.Variable(tf.truncated_normal([features_amount, hidden1_size], stddev=0.1))
b1 = tf.Variable(tf.constant(0.1, shape=[hidden1_size]))
z1 = tf.nn.relu(tf.matmul(x,W1)+b1)
W2 = tf.Variable(tf.truncated_normal([hidden1_size, hidden2_size], stddev=0.1))
b2 = tf.Variable(tf.constant(0.1, shape=[hidden2_size]))
z2 = tf.nn.relu(tf.matmul(z1,W2)+b2)
W3 = tf.Variable(tf.truncated_normal([hidden2_size, labels_amount], stddev=0.1))
b3 = tf.Variable(tf.constant(0.1, shape=[labels_amount]))
y = tf.nn.softmax(tf.matmul(z2, W3) + b3)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)


# W = tf.Variable(tf.zeros((features_amount, labels_amount)))
# b = tf.Variable(tf.zeros((labels_amount)))  # TODO change it to tf.random.uniform
# y = tf.nn.softmax(tf.matmul(x, W) + b)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=x)) #TODO try this

train_step = tf.train.GradientDescentOptimizer(learing_rate).minimize(cross_entropy)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
print(data)


def show_progress(i):
    print('Iteration:', i, ' loss:',cross_entropy.eval(session=sess, feed_dict={x: train_data, y_: train_states}))


for i in range(100000):
    sess.run(train_step, feed_dict={x: train_data, y_: train_states})  # BGD
    if i %100 ==0 :
        show_progress(i)


correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: test_data, y_: test_states}))




#Cannot feed value of shape (2175,) for Tensor 'Placeholder:0', which has shape '(?, 3)'